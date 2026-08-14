from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latentrouter.config import EvaluationConfig
from latentrouter.embedding.store import load_router_bundle
from latentrouter.evaluation.runner import evaluate_router
from latentrouter.experiments.model_contrastive_reread import (
    ModelContrastiveRereadBranch,
    PatchCache,
    PatchTokenStore,
    _subset_tensors,
    _top_pair,
)
from latentrouter.routers.base import BaseRouter
from latentrouter.routers.paper_latent_communication_router import (
    PaperSection32LatentCommunicationRouter,
    _CommTensors,
    _Section32LatentCommunicationNet,
)
from latentrouter.routers.paper_utils import _setting_cost_weight, _utility_targets
from latentrouter.schemas import RouterDatasetBundle


def _scored_state(net: Any, capsules: Any, models: Any, availability: Any) -> dict[str, Any]:
    raw = net.dist_head(models)
    mu = torch.sigmoid(raw[..., 0])
    sigma = torch.nn.functional.softplus(raw[..., 1]) + 1e-4
    pooled_capsules = capsules.mean(dim=1, keepdim=True).expand_as(models)
    delta_cap = net.residual_bound * torch.tanh(
        net.capsule_correction(torch.cat([models, pooled_capsules], dim=-1)).squeeze(-1)
    )
    mu_tilde = (mu + delta_cap).clamp(0.0, 1.0).masked_fill(~availability, -1e6)
    return {
        "mu": mu.masked_fill(~availability, -1e6),
        "sigma": sigma.masked_fill(~availability, 1.0),
        "delta_cap": delta_cap.masked_fill(~availability, 0.0),
        "mu_tilde": mu_tilde,
        "capsules": capsules,
        "model_tokens": models,
    }


def forward_joint_reread(
    router: Any,
    batch: _CommTensors,
    patches: Any,
    *,
    shuffle_contrast: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run an incomplete route, reread patches, and finish routing."""

    net = router.net_
    branch = router.reread_branch_
    query_tokens = net.query_token_proj(batch.query).view(
        batch.query.shape[0], net.num_query_tokens, net.hidden_dim
    )
    capsule_seed = net.capsule_queries.unsqueeze(0).expand(batch.query.shape[0], -1, -1)
    capsules, _ = net.capsule_init_attn(capsule_seed, query_tokens, query_tokens, need_weights=False)
    models = net.model_proj(batch.model_features)
    models = torch.where(batch.availability[:, :, None], models, torch.zeros_like(models))
    if len(net.layers) < 2:
        raise RuntimeError("Joint rereading requires at least two communication layers.")

    capsules, models, _ = net.layers[0](capsules, models, batch.availability)
    provisional = _scored_state(net, capsules, models, batch.availability)
    initial_route_scores = provisional["mu_tilde"] - _setting_cost_weight(router.setting) * batch.normalized_cost
    model_difference, score_gap = _top_pair(models, initial_route_scores, batch.availability)
    if shuffle_contrast and len(model_difference) > 1:
        model_difference = torch.roll(model_difference, shifts=1, dims=0)
        score_gap = torch.roll(score_gap, shifts=1, dims=0)

    capsules, reread_stats = branch(
        capsules,
        model_difference,
        score_gap,
        patches,
        mode="contrast",
    )
    for layer in net.layers[1:]:
        capsules, models, _ = layer(capsules, models, batch.availability)
    final = _scored_state(net, capsules, models, batch.availability)
    return final, provisional, reread_stats


class FromScratchModelContrastiveRouter(PaperSection32LatentCommunicationRouter):
    """LatentRouter and model-contrastive visual rereading trained jointly."""

    def __init__(
        self,
        model_ids: list[str],
        patch_cache: PatchCache,
        random_seed: int,
        *,
        initial_loss_weight: float = 0.30,
        attention_entropy_weight: float = 0.01,
        reread_bound: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_ids=model_ids,
            random_seed=random_seed,
            residual_bound=kwargs.pop("residual_bound", 0.05),
            **kwargs,
        )
        self.patch_cache = patch_cache
        self.initial_loss_weight = float(initial_loss_weight)
        self.attention_entropy_weight = float(attention_entropy_weight)
        self.reread_bound = float(reread_bound)
        self.reread_branch_: ModelContrastiveRereadBranch | None = None
        self.training_history_: list[dict[str, float | int]] = []

    def fit(self, train_bundle: RouterDatasetBundle, valid_bundle: RouterDatasetBundle | None = None) -> None:
        self._init_common(train_bundle)
        model_dim = int((self.profile_matrix_.shape[1] if self.profile_matrix_ is not None else 0) + 1)
        self.net_ = _Section32LatentCommunicationNet(
            query_dim=int(train_bundle.features.shape[1]),
            model_dim=model_dim,
            hidden_dim=self.hidden_dim,
            num_query_tokens=self.num_query_tokens,
            num_capsules=self.num_capsules,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            residual_bound=self.residual_bound,
            feedback_temperature=self.feedback_temperature,
        ).to(self.device_)
        patch_store = PatchTokenStore(self.patch_cache)
        self.reread_branch_ = ModelContrastiveRereadBranch(
            patch_dim=patch_store.patch_dim,
            hidden_dim=self.hidden_dim,
            num_capsules=self.num_capsules,
            reread_bound=self.reread_bound,
            zero_init_output=False,
        ).to(self.device_)
        self._train_joint(train_bundle, valid_bundle, patch_store)

    def _train_joint(
        self,
        train_bundle: RouterDatasetBundle,
        valid_bundle: RouterDatasetBundle | None,
        patch_store: PatchTokenStore,
    ) -> None:
        if self.net_ is None or self.reread_branch_ is None:
            raise RuntimeError("Joint model is not initialized.")
        device = torch.device(self.device_)
        rng = np.random.default_rng(self.random_seed)
        train = self._bundle_tensors(train_bundle)
        train_rows = patch_store.row_indices(train_bundle.sample_ids)
        valid = self._bundle_tensors(valid_bundle) if valid_bundle is not None else None
        valid_rows = patch_store.row_indices(valid_bundle.sample_ids) if valid_bundle is not None else None
        parameters = list(self.net_.parameters()) + list(self.reread_branch_.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        best_state: dict[str, dict[str, Any]] | None = None
        best_valid = -1e18
        stale = 0
        started = time.perf_counter()
        self.training_history_ = []

        for epoch in range(self.epochs):
            self.net_.train()
            self.reread_branch_.train()
            order = rng.permutation(len(train_bundle.sample_ids))
            epoch_total: list[float] = []
            epoch_final: list[float] = []
            epoch_initial: list[float] = []
            epoch_entropy: list[float] = []
            for batch_start in range(0, len(order), self.batch_size):
                positions = order[batch_start : batch_start + self.batch_size]
                index = torch.as_tensor(positions, dtype=torch.long, device=device)
                batch = _subset_tensors(train, index)
                patches = patch_store.batch(train_rows, positions, device)
                final, provisional, stats = forward_joint_reread(self, batch, patches)
                final_loss = self._loss(final, batch)
                initial_loss = self._loss(provisional, batch)
                normalized_entropy = stats["attention_entropy"].mean() / math.log(float(patch_store.patch_count))
                loss = (
                    final_loss
                    + self.initial_loss_weight * initial_loss
                    + self.attention_entropy_weight * normalized_entropy
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                epoch_total.append(float(loss.detach().cpu()))
                epoch_final.append(float(final_loss.detach().cpu()))
                epoch_initial.append(float(initial_loss.detach().cpu()))
                epoch_entropy.append(float(normalized_entropy.detach().cpu()))

            validation_selected_utility = float("nan")
            validation_loss = float("nan")
            if valid_bundle is not None and valid is not None and valid_rows is not None:
                self.net_.eval()
                self.reread_branch_.eval()
                with torch.no_grad():
                    positions = np.arange(len(valid_bundle.sample_ids), dtype=np.int64)
                    patches = patch_store.batch(valid_rows, positions, device)
                    final, _provisional, _stats = forward_joint_reread(self, valid, patches)
                    validation_loss = float(self._loss(final, valid).detach().cpu())
                    scores = final["mu_tilde"].detach().cpu().numpy().astype(np.float32)
                target = _utility_targets(valid_bundle, self.setting, self.cost_scale_)
                choices = np.where(valid_bundle.availability, scores, -1e6).argmax(axis=1)
                validation_selected_utility = float(np.nanmean(target[np.arange(len(choices)), choices]))
                selection_score = validation_selected_utility
            else:
                selection_score = -float(np.mean(epoch_total))

            self.training_history_.append({
                "epoch": epoch + 1,
                "loss": float(np.mean(epoch_total)),
                "final_route_loss": float(np.mean(epoch_final)),
                "initial_route_loss": float(np.mean(epoch_initial)),
                "normalized_attention_entropy": float(np.mean(epoch_entropy)),
                "validation_route_loss": validation_loss,
                "validation_selected_utility": validation_selected_utility,
            })
            if selection_score > best_valid + 1e-9:
                best_valid = selection_score
                best_state = {
                    "net": copy.deepcopy({k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}),
                    "branch": copy.deepcopy({k: v.detach().cpu().clone() for k, v in self.reread_branch_.state_dict().items()}),
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state["net"])
            self.reread_branch_.load_state_dict(best_state["branch"])
        self.fit_summary_ = {
            "router_name": self.__class__.__name__,
            "training": "joint_from_random_initialization",
            "epochs_requested": self.epochs,
            "epochs_completed": len(self.training_history_),
            "best_validation_selected_utility": best_valid,
            "training_time_seconds": float(time.perf_counter() - started),
            "initial_loss_weight": self.initial_loss_weight,
            "attention_entropy_weight": self.attention_entropy_weight,
            "reread_bound": self.reread_bound,
            "device": self.device_,
        }

    def predict_with_stats(
        self,
        bundle: RouterDatasetBundle,
        *,
        batch_size: int = 512,
        shuffle_contrast: bool = False,
    ) -> tuple[np.ndarray, dict[str, float], float]:
        if self.net_ is None or self.reread_branch_ is None:
            raise RuntimeError("Router must be fit before prediction.")
        self.net_.eval()
        self.reread_branch_.eval()
        patch_store = PatchTokenStore(self.patch_cache)
        rows = patch_store.row_indices(bundle.sample_ids)
        tensors = self._bundle_tensors(bundle)
        device = tensors.query.device
        predictions: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        peaks: list[np.ndarray] = []
        residuals: list[np.ndarray] = []
        started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                positions = np.arange(start, min(start + batch_size, len(rows)), dtype=np.int64)
                index = torch.as_tensor(positions, dtype=torch.long, device=device)
                batch = _subset_tensors(tensors, index)
                patches = patch_store.batch(rows, positions, device)
                final, _provisional, stats = forward_joint_reread(
                    self,
                    batch,
                    patches,
                    shuffle_contrast=shuffle_contrast,
                )
                predictions.append(final["mu_tilde"].detach().cpu().numpy().astype(np.float32))
                entropies.append(stats["attention_entropy"].detach().cpu().numpy())
                peaks.append(stats["attention_peak"].detach().cpu().numpy())
                residuals.append(stats["residual_norm"].detach().cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return np.concatenate(predictions), {
            "attention_entropy": float(np.mean(np.concatenate(entropies))),
            "attention_peak": float(np.mean(np.concatenate(peaks))),
            "residual_norm": float(np.mean(np.concatenate(residuals))),
        }, elapsed

    def predict_utilities(self, feature_bundle: RouterDatasetBundle, lambda_value: float | None = None) -> np.ndarray:
        del lambda_value
        scores, _stats, _elapsed = self.predict_with_stats(feature_bundle)
        scores[~feature_bundle.availability] = -1e6
        return scores


def run_from_scratch_experiment(
    *,
    processed_dir: str | Path,
    feature_dir: str | Path,
    patch_cache: PatchCache,
    output_dir: str | Path,
    seed: int,
    baseline_path: str | Path | None = None,
    epochs: int = 30,
    patience: int = 6,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    initial_loss_weight: float = 0.30,
    attention_entropy_weight: float = 0.01,
    reread_bound: float = 0.10,
    device: str = "cuda",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_bundle = load_router_bundle(processed_dir, split="train", feature_dir=feature_dir)
    valid_bundle = load_router_bundle(processed_dir, split="val", feature_dir=feature_dir)
    test_bundle = load_router_bundle(processed_dir, split="test", feature_dir=feature_dir)
    router = FromScratchModelContrastiveRouter(
        model_ids=list(train_bundle.model_ids),
        patch_cache=patch_cache,
        random_seed=seed,
        setting="performance_oriented",
        hidden_dim=48,
        num_query_tokens=4,
        num_capsules=7,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        residual_bound=0.05,
        feedback_temperature=0.25,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        calibration_fraction=0.35,
        max_calibration_samples=512,
        initial_loss_weight=initial_loss_weight,
        attention_entropy_weight=attention_entropy_weight,
        reread_bound=reread_bound,
        device=device,
    )
    router.fit(train_bundle, valid_bundle)
    router.save(output_dir / "router.pkl")
    scores, attention, prediction_time = router.predict_with_stats(test_bundle)
    shuffled_scores, shuffled_attention, shuffled_time = router.predict_with_stats(
        test_bundle, shuffle_contrast=True
    )
    evaluation = EvaluationConfig(
        lambda_min_exp=-4.0,
        lambda_max_exp=4.0,
        num_lambdas=51,
        output_dir=str(output_dir),
        aggregate_by=["dataset_name", "mode_id"],
        measure_throughput=True,
    )
    joint_metrics = evaluate_router(
        router_name="from_scratch_model_contrastive_reread",
        predicted_utilities=scores,
        bundle=test_bundle,
        config=evaluation,
        run_dir=output_dir / "joint",
        prediction_wall_time_seconds=prediction_time,
    )
    shuffled_metrics = evaluate_router(
        router_name="from_scratch_model_contrastive_reread_shuffled",
        predicted_utilities=shuffled_scores,
        bundle=test_bundle,
        config=evaluation,
        run_dir=output_dir / "joint_shuffled",
        prediction_wall_time_seconds=shuffled_time,
    )
    primary_metric_name = str(joint_metrics["metrics"].get("primary_metric", "nAUC"))
    joint_primary_metric = float(joint_metrics["metrics"]["primary_metric_value"])
    shuffled_primary_metric = float(shuffled_metrics["metrics"]["primary_metric_value"])
    baseline_nauc: float | None = None
    baseline_primary_metric: float | None = None
    baseline_prediction_time: float | None = None
    if baseline_path is not None and Path(baseline_path).exists():
        baseline = BaseRouter.load(baseline_path)
        baseline.device_ = router.device_
        baseline.net_.to(router.device_)
        started = time.perf_counter()
        baseline_scores = baseline.predict_utilities(test_bundle)
        if router.device_ == "cuda":
            torch.cuda.synchronize()
        baseline_prediction_time = time.perf_counter() - started
        baseline_metrics = evaluate_router(
            router_name="original_from_scratch_latentrouter",
            predicted_utilities=baseline_scores,
            bundle=test_bundle,
            config=evaluation,
            run_dir=output_dir / "baseline",
            prediction_wall_time_seconds=baseline_prediction_time,
        )
        baseline_primary_metric = float(baseline_metrics["metrics"]["primary_metric_value"])
        if "nAUC" in baseline_metrics["metrics"]:
            baseline_nauc = float(baseline_metrics["metrics"]["nAUC"])

    joint_nauc = float(joint_metrics["metrics"]["nAUC"]) if "nAUC" in joint_metrics["metrics"] else None
    shuffled_nauc = float(shuffled_metrics["metrics"]["nAUC"]) if "nAUC" in shuffled_metrics["metrics"] else None

    summary = {
        "seed": seed,
        "training": "all_router_and_reread_parameters_from_random_initialization",
        "epochs_completed": len(router.training_history_),
        "training_time_seconds": router.fit_summary_["training_time_seconds"],
        "primary_metric": primary_metric_name,
        "joint_primary_metric": joint_primary_metric,
        "shuffled_primary_metric": shuffled_primary_metric,
        "baseline_primary_metric": baseline_primary_metric,
        "joint_nAUC": joint_nauc,
        "shuffled_nAUC": shuffled_nauc,
        "shuffled_delta_vs_joint": float(
            shuffled_primary_metric - joint_primary_metric
        ),
        "baseline_nAUC": baseline_nauc,
        "delta_vs_baseline": None if baseline_primary_metric is None else float(joint_primary_metric - baseline_primary_metric),
        "prediction_time_seconds": prediction_time,
        "baseline_prediction_time_seconds": baseline_prediction_time,
        "attention": attention,
        "shuffled_attention": shuffled_attention,
        "fit_summary": router.fit_summary_,
        "history": router.training_history_,
    }
    np.save(output_dir / "joint_scores.npy", scores)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
