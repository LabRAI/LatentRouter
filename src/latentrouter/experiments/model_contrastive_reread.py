from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from latentrouter.config import EvaluationConfig
from latentrouter.embedding.store import load_router_bundle
from latentrouter.evaluation.runner import evaluate_router
from latentrouter.routers.base import BaseRouter
from latentrouter.routers.paper_latent_communication_router import _CommTensors
from latentrouter.schemas import RouterDatasetBundle

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - guarded at runtime.
    torch = None
    nn = None


@dataclass(slots=True)
class PatchCache:
    patches_path: Path
    sample_ids_path: Path
    manifest_path: Path


@dataclass(slots=True)
class RereadRunConfig:
    mode: str = "contrast"
    epochs: int = 20
    patience: int = 4
    batch_size: int = 128
    inference_batch_size: int = 512
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    entropy_weight: float = 0.0
    reread_bound: float = 0.10
    device: str = "cuda"


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the rereading experiment.")


def _sample_id_digest(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize_image_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist() if str(item)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
                if isinstance(decoded, list):
                    return [str(item) for item in decoded if str(item)]
            except json.JSONDecodeError:
                pass
        return [stripped]
    return [str(value)]


def build_openclip_patch_cache(
    *,
    processed_dir: str | Path,
    feature_dir: str | Path,
    cache_dir: str | Path,
    model_name: str = "ViT-B-32-quickgelu",
    pretrained: str = "openai",
    device: str = "cuda",
    batch_size: int = 64,
    force: bool = False,
) -> PatchCache:
    """Cache the final 7x7 OpenCLIP patch grid in feature-store sample order."""

    _require_torch()
    import pandas as pd
    from PIL import Image
    import open_clip

    processed_dir = Path(processed_dir)
    feature_dir = Path(feature_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    patches_path = cache_dir / "patches_fp16.npy"
    sample_ids_path = cache_dir / "sample_ids.json"
    manifest_path = cache_dir / "manifest.json"

    sample_index = pd.read_parquet(feature_dir / "sample_index.parquet")
    samples = pd.read_parquet(processed_dir / "samples.parquet")
    ordered = sample_index[["sample_id"]].merge(
        samples[["sample_id", "image_paths"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if ordered["image_paths"].isna().any():
        missing = int(ordered["image_paths"].isna().sum())
        raise ValueError(f"Patch extraction is missing image paths for {missing} samples.")
    sample_ids = ordered["sample_id"].astype(str).tolist()
    digest = _sample_id_digest(sample_ids)

    if not force and patches_path.exists() and sample_ids_path.exists() and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("sample_id_sha256") == digest and int(manifest.get("sample_count", -1)) == len(sample_ids):
            return PatchCache(patches_path, sample_ids_path, manifest_path)

    target_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for patch extraction but is unavailable.")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=target_device,
    )
    model.eval()
    model.visual.output_tokens = True

    probe_path = _normalize_image_paths(ordered.iloc[0]["image_paths"])[0]
    with Image.open(probe_path) as image:
        probe = preprocess(image.convert("RGB")).unsqueeze(0).to(target_device)
    with torch.no_grad():
        probe_output = model.visual(probe)
    if not isinstance(probe_output, tuple) or len(probe_output) != 2:
        raise RuntimeError(f"OpenCLIP visual encoder did not return pooled and patch tokens: {type(probe_output)!r}")
    _, probe_tokens = probe_output
    patch_count = int(probe_tokens.shape[1])
    patch_dim = int(probe_tokens.shape[2])

    temporary_path = patches_path.with_suffix(".tmp.npy")
    mmap = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(sample_ids), patch_count, patch_dim),
    )
    missing_images = 0
    started = time.perf_counter()
    amp_enabled = target_device.type == "cuda"

    try:
        for start in range(0, len(ordered), batch_size):
            stop = min(start + batch_size, len(ordered))
            tensors: list[Any] = []
            valid_positions: list[int] = []
            for local_index, value in enumerate(ordered.iloc[start:stop]["image_paths"].tolist()):
                paths = _normalize_image_paths(value)
                image_path = next((candidate for candidate in paths if Path(candidate).exists()), None)
                if image_path is None:
                    missing_images += 1
                    continue
                try:
                    with Image.open(image_path) as image:
                        tensors.append(preprocess(image.convert("RGB")))
                    valid_positions.append(local_index)
                except Exception:
                    missing_images += 1
            batch_array = np.zeros((stop - start, patch_count, patch_dim), dtype=np.float16)
            if tensors:
                inputs = torch.stack(tensors, dim=0).to(target_device, non_blocking=True)
                with torch.no_grad(), torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model.visual(inputs)
                    _, patch_tokens = output
                values = patch_tokens.detach().float().cpu().numpy().astype(np.float16)
                batch_array[np.asarray(valid_positions, dtype=np.int64)] = values
            mmap[start:stop] = batch_array
            if target_device.type == "cuda":
                torch.cuda.synchronize()
        mmap.flush()
    finally:
        del mmap

    os.replace(temporary_path, patches_path)
    with sample_ids_path.open("w", encoding="utf-8") as handle:
        json.dump(sample_ids, handle)
    elapsed = time.perf_counter() - started
    manifest = {
        "model_name": model_name,
        "pretrained": pretrained,
        "sample_count": len(sample_ids),
        "patch_count": patch_count,
        "patch_dim": patch_dim,
        "dtype": "float16",
        "sample_id_sha256": digest,
        "missing_images": missing_images,
        "extraction_time_seconds": elapsed,
        "device": str(target_device),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return PatchCache(patches_path, sample_ids_path, manifest_path)


class PatchTokenStore:
    def __init__(self, cache: PatchCache) -> None:
        with cache.sample_ids_path.open("r", encoding="utf-8") as handle:
            sample_ids = [str(item) for item in json.load(handle)]
        self.patches = np.load(cache.patches_path, mmap_mode="r")
        if len(sample_ids) != len(self.patches):
            raise ValueError("Patch cache sample id count does not match the patch array.")
        self.index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        if len(self.index) != len(sample_ids):
            raise ValueError("Patch cache contains duplicate sample ids.")

    @property
    def patch_count(self) -> int:
        return int(self.patches.shape[1])

    @property
    def patch_dim(self) -> int:
        return int(self.patches.shape[2])

    def row_indices(self, sample_ids: np.ndarray) -> np.ndarray:
        try:
            return np.asarray([self.index[str(sample_id)] for sample_id in sample_ids], dtype=np.int64)
        except KeyError as error:
            raise KeyError(f"Sample id is absent from the patch cache: {error.args[0]}") from error

    def batch(self, row_indices: np.ndarray, positions: np.ndarray, device: Any) -> Any:
        selected = np.asarray(self.patches[row_indices[positions]], dtype=np.float32)
        return torch.from_numpy(selected).to(device, non_blocking=True)


class ModelContrastiveRereadBranch(nn.Module if nn is not None else object):
    """One-query spatial reread residual inserted between communication blocks."""

    def __init__(
        self,
        patch_dim: int,
        hidden_dim: int,
        num_capsules: int = 7,
        reread_bound: float = 0.10,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_capsules = int(num_capsules)
        self.reread_bound = float(reread_bound)
        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_key = nn.Linear(patch_dim, hidden_dim, bias=False)
        self.patch_value = nn.Linear(patch_dim, hidden_dim, bias=False)
        self.model_contrast_query = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.capsule_query = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.evidence_out = nn.Linear(hidden_dim, hidden_dim * num_capsules, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.evidence_out.weight)
        else:
            nn.init.xavier_uniform_(self.evidence_out.weight, gain=0.1)

    def forward(
        self,
        capsules: Any,
        model_difference: Any,
        score_gap: Any,
        patches: Any,
        *,
        mode: str,
    ) -> tuple[Any, dict[str, Any]]:
        if mode not in {"contrast", "generic"}:
            raise ValueError(f"Unsupported reread mode: {mode}")
        capsule_summary = capsules.mean(dim=1)
        capsule_query = self.capsule_query(capsule_summary)
        if mode == "contrast":
            contrast_input = torch.cat([model_difference, score_gap[:, None]], dim=-1)
            query = self.query_norm(self.model_contrast_query(contrast_input) + 0.25 * capsule_query)
        else:
            query = self.query_norm(capsule_query)
        normalized_patches = self.patch_norm(patches)
        keys = self.patch_key(normalized_patches)
        values = self.patch_value(normalized_patches)
        logits = torch.einsum("bd,bpd->bp", query, keys) / math.sqrt(float(self.hidden_dim))
        attention = torch.softmax(logits, dim=-1)
        evidence = torch.einsum("bp,bpd->bd", attention, values)
        residual = self.reread_bound * torch.tanh(self.evidence_out(evidence))
        residual = residual.view(len(capsules), self.num_capsules, self.hidden_dim)
        updated_capsules = capsules + residual
        entropy = -(attention.clamp_min(1e-8) * torch.log(attention.clamp_min(1e-8))).sum(dim=-1)
        return updated_capsules, {
            "attention_entropy": entropy,
            "attention_peak": attention.max(dim=-1).values,
            "residual_norm": residual.flatten(1).norm(dim=-1),
        }


def _subset_tensors(tensors: _CommTensors, index: Any) -> _CommTensors:
    return _CommTensors(
        query=tensors.query[index],
        model_features=tensors.model_features[index],
        availability=tensors.availability[index],
        quality=tensors.quality[index],
        utility=tensors.utility[index],
        normalized_cost=tensors.normalized_cost[index],
    )


def _top_pair(models: Any, provisional_scores: Any, availability: Any) -> tuple[Any, Any]:
    masked = provisional_scores.masked_fill(~availability, -1e9)
    top_indices = torch.topk(masked, k=2, dim=1).indices
    gather_index = top_indices[..., None].expand(-1, -1, models.shape[-1])
    contenders = torch.gather(models, dim=1, index=gather_index)
    contender_scores = torch.gather(masked, dim=1, index=top_indices)
    difference = contenders[:, 0] - contenders[:, 1]
    gap = contender_scores[:, 0] - contender_scores[:, 1]
    return difference, gap


def forward_with_reread(
    base_router: Any,
    branch: ModelContrastiveRereadBranch,
    batch: _CommTensors,
    patches: Any,
    *,
    mode: str,
    shuffle_contrast: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    net = base_router.net_
    query_tokens = net.query_token_proj(batch.query).view(
        batch.query.shape[0], net.num_query_tokens, net.hidden_dim
    )
    capsule_seed = net.capsule_queries.unsqueeze(0).expand(batch.query.shape[0], -1, -1)
    capsules, _ = net.capsule_init_attn(capsule_seed, query_tokens, query_tokens, need_weights=False)
    models = net.model_proj(batch.model_features)
    models = torch.where(batch.availability[:, :, None], models, torch.zeros_like(models))

    if not net.layers:
        raise RuntimeError("The baseline network has no communication layers.")
    capsules, models, _ = net.layers[0](capsules, models, batch.availability)
    provisional_raw = net.dist_head(models)
    provisional_scores = torch.sigmoid(provisional_raw[..., 0])
    model_difference, score_gap = _top_pair(models, provisional_scores, batch.availability)
    if shuffle_contrast and len(model_difference) > 1:
        model_difference = torch.roll(model_difference, shifts=1, dims=0)
        score_gap = torch.roll(score_gap, shifts=1, dims=0)
    capsules, reread_stats = branch(
        capsules,
        model_difference,
        score_gap,
        patches,
        mode=mode,
    )
    for layer in net.layers[1:]:
        capsules, models, _ = layer(capsules, models, batch.availability)

    raw = net.dist_head(models)
    mu = torch.sigmoid(raw[..., 0])
    sigma = torch.nn.functional.softplus(raw[..., 1]) + 1e-4
    pooled_capsules = capsules.mean(dim=1, keepdim=True).expand_as(models)
    delta_cap = net.residual_bound * torch.tanh(
        net.capsule_correction(torch.cat([models, pooled_capsules], dim=-1)).squeeze(-1)
    )
    mu_tilde = (mu + delta_cap).clamp(0.0, 1.0).masked_fill(~batch.availability, -1e6)
    return {
        "mu": mu.masked_fill(~batch.availability, -1e6),
        "sigma": sigma.masked_fill(~batch.availability, 1.0),
        "delta_cap": delta_cap.masked_fill(~batch.availability, 0.0),
        "mu_tilde": mu_tilde,
        "capsules": capsules,
        "model_tokens": models,
    }, reread_stats


def _predict(
    *,
    base_router: Any,
    branch: ModelContrastiveRereadBranch,
    tensors: _CommTensors,
    patch_store: PatchTokenStore,
    patch_rows: np.ndarray,
    batch_size: int,
    mode: str,
    shuffle_contrast: bool = False,
) -> tuple[np.ndarray, dict[str, float], float]:
    base_router.net_.eval()
    branch.eval()
    predictions: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    peaks: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    device = tensors.query.device
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(patch_rows), batch_size):
            positions = np.arange(start, min(start + batch_size, len(patch_rows)), dtype=np.int64)
            index = torch.as_tensor(positions, dtype=torch.long, device=device)
            batch = _subset_tensors(tensors, index)
            patches = patch_store.batch(patch_rows, positions, device)
            out, stats = forward_with_reread(
                base_router,
                branch,
                batch,
                patches,
                mode=mode,
                shuffle_contrast=shuffle_contrast,
            )
            predictions.append(out["mu_tilde"].detach().cpu().numpy().astype(np.float32))
            entropies.append(stats["attention_entropy"].detach().cpu().numpy())
            peaks.append(stats["attention_peak"].detach().cpu().numpy())
            residual_norms.append(stats["residual_norm"].detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return np.concatenate(predictions, axis=0), {
        "attention_entropy": float(np.mean(np.concatenate(entropies))),
        "attention_peak": float(np.mean(np.concatenate(peaks))),
        "residual_norm": float(np.mean(np.concatenate(residual_norms))),
    }, elapsed


def _selected_quality(scores: np.ndarray, bundle: RouterDatasetBundle) -> float:
    selected = np.where(bundle.availability, scores, -1e6).argmax(axis=1)
    quality = bundle.correctness[np.arange(len(selected)), selected]
    return float(np.nanmean(quality))


def _flip_diagnostics(
    base_scores: np.ndarray,
    reread_scores: np.ndarray,
    bundle: RouterDatasetBundle,
    close_threshold: float = 0.02,
) -> dict[str, float | int]:
    masked_base = np.where(bundle.availability, base_scores, -1e6)
    masked_reread = np.where(bundle.availability, reread_scores, -1e6)
    base_choice = masked_base.argmax(axis=1)
    reread_choice = masked_reread.argmax(axis=1)
    rows = np.arange(len(base_choice))
    base_quality = bundle.correctness[rows, base_choice]
    reread_quality = bundle.correctness[rows, reread_choice]
    changed = base_choice != reread_choice
    helpful = reread_quality > base_quality
    harmful = reread_quality < base_quality
    sorted_scores = np.sort(masked_base, axis=1)
    close = (sorted_scores[:, -1] - sorted_scores[:, -2]) <= close_threshold
    return {
        "changed_count": int(changed.sum()),
        "changed_rate": float(changed.mean()),
        "helpful_flip_count": int((changed & helpful).sum()),
        "helpful_flip_rate": float((changed & helpful).mean()),
        "harmful_flip_count": int((changed & harmful).sum()),
        "harmful_flip_rate": float((changed & harmful).mean()),
        "net_helpful_flip_rate": float(((changed & helpful).sum() - (changed & harmful).sum()) / len(changed)),
        "close_sample_count": int(close.sum()),
        "close_helpful_flip_rate": float((close & changed & helpful).sum() / max(int(close.sum()), 1)),
        "close_harmful_flip_rate": float((close & changed & harmful).sum() / max(int(close.sum()), 1)),
        "base_selected_quality": float(np.nanmean(base_quality)),
        "reread_selected_quality": float(np.nanmean(reread_quality)),
    }


def run_reread_experiment(
    *,
    baseline_path: str | Path,
    processed_dir: str | Path,
    feature_dir: str | Path,
    patch_cache: PatchCache,
    output_dir: str | Path,
    seed: int,
    config: RereadRunConfig,
) -> dict[str, Any]:
    _require_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_bundle = load_router_bundle(processed_dir, split="train", feature_dir=feature_dir)
    valid_bundle = load_router_bundle(processed_dir, split="val", feature_dir=feature_dir)
    test_bundle = load_router_bundle(processed_dir, split="test", feature_dir=feature_dir)
    patch_store = PatchTokenStore(patch_cache)
    train_rows = patch_store.row_indices(train_bundle.sample_ids)
    valid_rows = patch_store.row_indices(valid_bundle.sample_ids)
    test_rows = patch_store.row_indices(test_bundle.sample_ids)

    base_router = BaseRouter.load(baseline_path)
    if not hasattr(base_router, "net_") or base_router.net_ is None:
        raise TypeError("The baseline checkpoint is not a fitted latent communication router.")
    base_router.device_ = str(target_device)
    base_router.net_.to(target_device)
    base_router.net_.eval()
    for parameter in base_router.net_.parameters():
        parameter.requires_grad_(False)

    branch = ModelContrastiveRereadBranch(
        patch_dim=patch_store.patch_dim,
        hidden_dim=int(base_router.hidden_dim),
        num_capsules=int(base_router.num_capsules),
        reread_bound=config.reread_bound,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        branch.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_tensors = base_router._bundle_tensors(train_bundle)
    valid_tensors = base_router._bundle_tensors(valid_bundle)
    rng = np.random.default_rng(seed)
    best_state: dict[str, Any] | None = None
    best_validation_quality = -1e18
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        branch.train()
        base_router.net_.eval()
        order = rng.permutation(len(train_bundle.sample_ids))
        losses: list[float] = []
        route_losses: list[float] = []
        entropy_losses: list[float] = []
        for batch_start in range(0, len(order), config.batch_size):
            positions = order[batch_start : batch_start + config.batch_size]
            if len(positions) == 0:
                continue
            index = torch.as_tensor(positions, dtype=torch.long, device=target_device)
            batch = _subset_tensors(train_tensors, index)
            patches = patch_store.batch(train_rows, positions, target_device)
            out, stats = forward_with_reread(
                base_router,
                branch,
                batch,
                patches,
                mode=config.mode,
            )
            route_loss = base_router._loss(out, batch)
            normalized_entropy = stats["attention_entropy"].mean() / math.log(float(patch_store.patch_count))
            loss = route_loss + config.entropy_weight * normalized_entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            route_losses.append(float(route_loss.detach().cpu()))
            entropy_losses.append(float(normalized_entropy.detach().cpu()))

        branch.eval()
        valid_positions = np.arange(len(valid_rows), dtype=np.int64)
        valid_patches = patch_store.batch(valid_rows, valid_positions, target_device)
        with torch.no_grad():
            valid_out, valid_forward_stats = forward_with_reread(
                base_router,
                branch,
                valid_tensors,
                valid_patches,
                mode=config.mode,
            )
            valid_route_loss = float(base_router._loss(valid_out, valid_tensors).detach().cpu())
            valid_scores = valid_out["mu_tilde"].detach().cpu().numpy().astype(np.float32)
            valid_stats = {
                "attention_peak": float(valid_forward_stats["attention_peak"].mean().detach().cpu()),
                "residual_norm": float(valid_forward_stats["residual_norm"].mean().detach().cpu()),
            }
        validation_quality = _selected_quality(valid_scores, valid_bundle)
        history.append({
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "route_loss": float(np.mean(route_losses)),
            "normalized_attention_entropy": float(np.mean(entropy_losses)),
            "validation_selected_quality": validation_quality,
            "validation_route_loss": valid_route_loss,
            "validation_attention_peak": valid_stats["attention_peak"],
            "validation_residual_norm": valid_stats["residual_norm"],
        })
        validation_score = -valid_route_loss
        if validation_score > best_validation_quality + 1e-9:
            best_validation_quality = validation_score
            best_state = copy.deepcopy({key: value.detach().cpu().clone() for key, value in branch.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        branch.load_state_dict(best_state)
    training_time = time.perf_counter() - started
    test_tensors = base_router._bundle_tensors(test_bundle)
    reread_scores, attention_stats, prediction_time = _predict(
        base_router=base_router,
        branch=branch,
        tensors=test_tensors,
        patch_store=patch_store,
        patch_rows=test_rows,
        batch_size=config.inference_batch_size,
        mode=config.mode,
    )
    shuffled_scores, shuffled_attention_stats, shuffled_prediction_time = _predict(
        base_router=base_router,
        branch=branch,
        tensors=test_tensors,
        patch_store=patch_store,
        patch_rows=test_rows,
        batch_size=config.inference_batch_size,
        mode=config.mode,
        shuffle_contrast=config.mode == "contrast",
    )
    base_started = time.perf_counter()
    base_scores = base_router.predict_utilities(test_bundle)
    if target_device.type == "cuda":
        torch.cuda.synchronize()
    base_prediction_time = time.perf_counter() - base_started

    evaluation = EvaluationConfig(
        lambda_min_exp=-4.0,
        lambda_max_exp=4.0,
        num_lambdas=51,
        output_dir=str(output_dir),
        aggregate_by=["dataset_name", "mode_id"],
        measure_throughput=True,
    )
    baseline_metrics = evaluate_router(
        router_name="latentrouter_baseline",
        predicted_utilities=base_scores,
        bundle=test_bundle,
        config=evaluation,
        run_dir=output_dir / "baseline",
        prediction_wall_time_seconds=base_prediction_time,
    )
    reread_metrics = evaluate_router(
        router_name=f"model_contrastive_reread_{config.mode}",
        predicted_utilities=reread_scores,
        bundle=test_bundle,
        config=evaluation,
        run_dir=output_dir / config.mode,
        prediction_wall_time_seconds=prediction_time,
    )
    shuffled_metrics = evaluate_router(
        router_name=f"model_contrastive_reread_{config.mode}_shuffled",
        predicted_utilities=shuffled_scores,
        bundle=test_bundle,
        config=evaluation,
        run_dir=output_dir / f"{config.mode}_shuffled",
        prediction_wall_time_seconds=shuffled_prediction_time,
    )
    baseline_metric_values = baseline_metrics["metrics"]
    reread_metric_values = reread_metrics["metrics"]
    shuffled_metric_values = shuffled_metrics["metrics"]

    np.save(output_dir / "reread_scores.npy", reread_scores)
    np.save(output_dir / "base_scores.npy", base_scores)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in branch.state_dict().items()},
            "patch_dim": patch_store.patch_dim,
            "patch_count": patch_store.patch_count,
            "hidden_dim": int(base_router.hidden_dim),
            "num_capsules": int(base_router.num_capsules),
            "config": config.__dict__ if hasattr(config, "__dict__") else {
                field: getattr(config, field) for field in config.__dataclass_fields__
            },
        },
        output_dir / "branch.pt",
    )
    summary = {
        "seed": seed,
        "mode": config.mode,
        "baseline_path": str(Path(baseline_path)),
        "patch_count": patch_store.patch_count,
        "patch_dim": patch_store.patch_dim,
        "epochs_completed": len(history),
        "best_validation_negative_route_loss": best_validation_quality,
        "training_time_seconds": training_time,
        "base_prediction_time_seconds": base_prediction_time,
        "reread_prediction_time_seconds": prediction_time,
        "prediction_slowdown": prediction_time / max(base_prediction_time, 1e-9),
        "baseline_nAUC": float(baseline_metric_values["nAUC"]),
        "reread_nAUC": float(reread_metric_values["nAUC"]),
        "nAUC_delta": float(reread_metric_values["nAUC"] - baseline_metric_values["nAUC"]),
        "shuffled_nAUC": float(shuffled_metric_values["nAUC"]),
        "shuffled_nAUC_delta_vs_reread": float(shuffled_metric_values["nAUC"] - reread_metric_values["nAUC"]),
        "attention": attention_stats,
        "shuffled_attention": shuffled_attention_stats,
        "flips": _flip_diagnostics(base_scores, reread_scores, test_bundle),
        "history": history,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
