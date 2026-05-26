from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from latentrouter.routers.base import BaseRouter
from latentrouter.routers.registry import register_router
from latentrouter.schemas import RouterDatasetBundle

from .paper_utils import (
    TORCH_AVAILABLE,
    _device_name,
    _masked_gaussian_nll,
    _masked_listwise_loss,
    _masked_pairwise_loss,
    _masked_regression_loss,
    _normalized_cost,
    _profile_matrix,
    _setting_cost_weight,
    _utility_targets,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - runtime guard.
    torch = None
    nn = None
    F = None


class _Section32CommunicationLayer(nn.Module if TORCH_AVAILABLE else object):
    """Paper Section 3.2 latent communication block.

    The layer performs model-conditioned capsule reads, pairwise model-token
    communication with availability masking, and decision-aware capsule feedback.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, feedback_temperature: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_dim // num_heads)
        self.feedback_temperature = float(feedback_temperature)
        self.dropout = nn.Dropout(float(dropout))

        self.cap_key = nn.Linear(hidden_dim, hidden_dim)
        self.cap_value = nn.Linear(hidden_dim, hidden_dim)
        self.model_query = nn.Linear(hidden_dim, hidden_dim)
        self.model_read_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.pair_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.pair_bias = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, num_heads),
        )
        self.pair_out = nn.Linear(hidden_dim, hidden_dim)

        self.feedback_score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.feedback_q = nn.Linear(hidden_dim, hidden_dim)
        self.feedback_k = nn.Linear(hidden_dim, hidden_dim)
        self.feedback_v = nn.Linear(hidden_dim, hidden_dim)
        self.feedback_out = nn.Linear(hidden_dim, hidden_dim)

        self.model_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.capsule_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_model_read = nn.LayerNorm(hidden_dim)
        self.norm_model_pair = nn.LayerNorm(hidden_dim)
        self.norm_model_ffn = nn.LayerNorm(hidden_dim)
        self.norm_capsule_feedback = nn.LayerNorm(hidden_dim)
        self.norm_capsule_ffn = nn.LayerNorm(hidden_dim)

    def _split_heads(self, x: Any) -> Any:
        batch, length, _dim = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, capsules: Any, models: Any, availability: Any) -> tuple[Any, Any, dict[str, Any]]:
        stats: dict[str, Any] = {}
        available = availability.bool()

        # 1. Each model token reads routing capsules from its own capability perspective.
        q = self.model_query(models)
        k = self.cap_key(capsules)
        v = self.cap_value(capsules)
        read_logits = torch.einsum("bkd,bcd->bkc", q, k) / math.sqrt(float(self.hidden_dim))
        read_attn = torch.softmax(read_logits, dim=-1)
        cap_context = torch.einsum("bkc,bcd->bkd", read_attn, v)
        read_delta = self.model_read_update(torch.cat([models, cap_context, models * cap_context], dim=-1))
        models = self.norm_model_read(models + self.dropout(read_delta))
        models = torch.where(available[:, :, None], models, torch.zeros_like(models))
        stats["capsule_read_entropy"] = -(read_attn.clamp_min(1e-8) * torch.log(read_attn.clamp_min(1e-8))).sum(dim=-1).mean()

        # 2. Candidate model tokens compare through availability-masked pairwise communication.
        qkv = self.pair_qkv(models)
        q_raw, k_raw, v_raw = qkv.chunk(3, dim=-1)
        qh = self._split_heads(q_raw)
        kh = self._split_heads(k_raw)
        vh = self._split_heads(v_raw)
        pair_logits = torch.einsum("bhid,bhjd->bhij", qh, kh) / math.sqrt(float(self.head_dim))
        mi = models[:, :, None, :].expand(-1, -1, models.shape[1], -1)
        mj = models[:, None, :, :].expand(-1, models.shape[1], -1, -1)
        bias = self.pair_bias(torch.cat([mi, mj, mi - mj, mi * mj], dim=-1)).permute(0, 3, 1, 2)
        pair_logits = pair_logits + bias
        pair_logits = pair_logits.masked_fill(~available[:, None, None, :], -1e9)
        pair_attn = torch.softmax(pair_logits, dim=-1)
        pair_context = torch.einsum("bhij,bhjd->bhid", pair_attn, vh).transpose(1, 2).contiguous()
        pair_context = pair_context.view(models.shape[0], models.shape[1], self.hidden_dim)
        pair_delta = self.pair_out(pair_context)
        models = self.norm_model_pair(models + self.dropout(pair_delta))
        models = self.norm_model_ffn(models + self.dropout(self.model_ffn(models)))
        models = torch.where(available[:, :, None], models, torch.zeros_like(models))
        stats["pairwise_attention_entropy"] = -(pair_attn.clamp_min(1e-8) * torch.log(pair_attn.clamp_min(1e-8))).sum(dim=-1).mean()

        # 3. Routing capsules receive feedback from decision-relevant model states.
        current_scores = self.feedback_score(models).squeeze(-1).masked_fill(~available, -1e9)
        weights = torch.softmax(current_scores / max(self.feedback_temperature, 1e-6), dim=1)
        weighted_models = models * weights[:, :, None]
        fq = self._split_heads(self.feedback_q(capsules))
        fk = self._split_heads(self.feedback_k(weighted_models))
        fv = self._split_heads(self.feedback_v(weighted_models))
        feedback_logits = torch.einsum("bhcd,bhkd->bhck", fq, fk) / math.sqrt(float(self.head_dim))
        feedback_logits = feedback_logits.masked_fill(~available[:, None, None, :], -1e9)
        feedback_attn = torch.softmax(feedback_logits, dim=-1)
        feedback_context = torch.einsum("bhck,bhkd->bhcd", feedback_attn, fv).transpose(1, 2).contiguous()
        feedback_context = feedback_context.view(capsules.shape[0], capsules.shape[1], self.hidden_dim)
        capsules = self.norm_capsule_feedback(capsules + self.dropout(self.feedback_out(feedback_context)))
        capsules = self.norm_capsule_ffn(capsules + self.dropout(self.capsule_ffn(capsules)))
        stats["feedback_entropy"] = -(feedback_attn.clamp_min(1e-8) * torch.log(feedback_attn.clamp_min(1e-8))).sum(dim=-1).mean()
        return capsules, models, stats


class _Section32LatentCommunicationNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(
        self,
        query_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_query_tokens: int,
        num_capsules: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        residual_bound: float,
        feedback_temperature: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.num_capsules = int(num_capsules)
        self.residual_bound = float(residual_bound)
        self.query_token_proj = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim * num_query_tokens),
        )
        self.capsule_queries = nn.Parameter(torch.randn(num_capsules, hidden_dim) * 0.02)
        self.capsule_init_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=float(dropout), batch_first=True)
        self.model_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                _Section32CommunicationLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    feedback_temperature=feedback_temperature,
                )
                for _ in range(num_layers)
            ]
        )
        self.dist_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 2),
        )
        self.capsule_correction = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: Any, model_features: Any, availability: Any) -> dict[str, Any]:
        batch = query.shape[0]
        query_tokens = self.query_token_proj(query).view(batch, self.num_query_tokens, self.hidden_dim)
        capsule_seed = self.capsule_queries.unsqueeze(0).expand(batch, -1, -1)
        capsules, _ = self.capsule_init_attn(capsule_seed, query_tokens, query_tokens, need_weights=False)
        models = self.model_proj(model_features)
        models = torch.where(availability.bool()[:, :, None], models, torch.zeros_like(models))
        layer_stats: list[dict[str, Any]] = []
        for layer in self.layers:
            capsules, models, stats = layer(capsules, models, availability)
            layer_stats.append(stats)
        raw = self.dist_head(models)
        mu = torch.sigmoid(raw[..., 0])
        sigma = F.softplus(raw[..., 1]) + 1e-4
        pooled_capsules = capsules.mean(dim=1, keepdim=True).expand_as(models)
        delta_cap = self.residual_bound * torch.tanh(
            self.capsule_correction(torch.cat([models, pooled_capsules], dim=-1)).squeeze(-1)
        )
        mu_tilde = (mu + delta_cap).clamp(0.0, 1.0).masked_fill(~availability.bool(), -1e6)
        return {
            "mu": mu.masked_fill(~availability.bool(), -1e6),
            "sigma": sigma.masked_fill(~availability.bool(), 1.0),
            "delta_cap": delta_cap.masked_fill(~availability.bool(), 0.0),
            "mu_tilde": mu_tilde,
            "capsules": capsules,
            "model_tokens": models,
            "layer_stats": layer_stats,
        }


@dataclass
class _CommTensors:
    query: Any
    model_features: Any
    availability: Any
    quality: Any
    utility: Any
    normalized_cost: Any


@register_router("paper_section_3_2_latent_communication")
class PaperSection32LatentCommunicationRouter(BaseRouter):
    """Trainable implementation of the latent communication block in Section 3.2.

    This router is intentionally separate from the accepted artifact-composition
    R4.3 policy. It lets us test whether the paper's explicit capsule/model-token
    communication can strengthen the current fair-hashing LatentRouter line.
    """

    def __init__(
        self,
        model_ids: list[str],
        random_seed: int = 20260308,
        *,
        setting: str = "performance_oriented",
        hidden_dim: int = 96,
        num_query_tokens: int = 6,
        num_capsules: int = 7,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_bound: float = 0.05,
        feedback_temperature: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 16,
        patience: int = 4,
        calibration_fraction: float = 0.35,
        max_calibration_samples: int | None = 4096,
        memory_encoder_type: str = "profile_mlp_pairwise",
        dist_weight: float = 1.0,
        pairwise_weight: float = 0.35,
        listwise_weight: float = 0.65,
        utility_weight: float = 0.25,
        residual_weight: float = 0.05,
        mean_weight: float = 0.10,
        device: str = "auto",
        **kwargs: object,
    ) -> None:
        super().__init__(model_ids=model_ids, random_seed=random_seed, **kwargs)
        self.setting = str(setting)
        self.hidden_dim = int(hidden_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.num_capsules = int(num_capsules)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.residual_bound = float(residual_bound)
        self.feedback_temperature = float(feedback_temperature)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.calibration_fraction = float(calibration_fraction)
        self.max_calibration_samples = None if max_calibration_samples is None else int(max_calibration_samples)
        self.memory_encoder_type = str(memory_encoder_type)
        self.dist_weight = float(dist_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.listwise_weight = float(listwise_weight)
        self.utility_weight = float(utility_weight)
        self.residual_weight = float(residual_weight)
        self.mean_weight = float(mean_weight)
        self.device_requested = str(device)
        self.query_scaler_: StandardScaler | None = None
        self.profile_scaler_: StandardScaler | None = None
        self.profile_matrix_: np.ndarray | None = None
        self.profile_feature_names_: list[str] = []
        self.profile_summary_: dict[str, Any] = {}
        self.cost_scale_: float = 1.0
        self.net_: Any | None = None
        self.device_: str = "cpu"
        self.fit_summary_: dict[str, Any] = {}

    def _init_common(self, train_bundle: RouterDatasetBundle) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for paper_section_3_2_latent_communication.")
        self.device_ = _device_name(self.device_requested)
        seed = int(self.random_seed)
        torch.manual_seed(seed)
        if self.device_ == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.query_scaler_ = StandardScaler().fit(train_bundle.features)
        raw_profile, names, summary = _profile_matrix(
            train_bundle,
            self.model_ids,
            seed=seed,
            encoder_type=self.memory_encoder_type,
            calibration_fraction=self.calibration_fraction,
            max_calibration_samples=self.max_calibration_samples,
        )
        self.profile_feature_names_ = names
        self.profile_scaler_ = StandardScaler().fit(raw_profile)
        self.profile_matrix_ = self.profile_scaler_.transform(raw_profile).astype(np.float32)
        self.profile_summary_ = {
            "encoder_type": self.memory_encoder_type,
            "profile_dim": int(raw_profile.shape[1]),
            "calibration_ids_count": len(summary.get("calibration_ids", [])),
            "train_only_profiles": True,
        }
        availability = train_bundle.availability.astype(bool, copy=False)
        self.cost_scale_ = float(np.nanmax(train_bundle.costs[availability])) if np.any(availability) else 1.0

    def _make_model_features(self, bundle: RouterDatasetBundle) -> np.ndarray:
        if self.profile_matrix_ is None:
            raise RuntimeError("Router must be fit before model features are available.")
        availability = bundle.availability.astype(bool, copy=False)
        profile = np.repeat(self.profile_matrix_[None, :, :], len(bundle.sample_ids), axis=0)
        cost = _normalized_cost(bundle, self.cost_scale_)[..., None]
        features = np.concatenate([profile, cost], axis=2).astype(np.float32)
        features[~availability] = 0.0
        return features

    def _bundle_tensors(self, bundle: RouterDatasetBundle) -> _CommTensors:
        if self.query_scaler_ is None:
            raise RuntimeError("Router must be fit before tensor conversion.")
        device = torch.device(self.device_)
        query = self.query_scaler_.transform(bundle.features).astype(np.float32)
        model_features = self._make_model_features(bundle)
        utility = _utility_targets(bundle, self.setting, self.cost_scale_)
        normalized_cost = _normalized_cost(bundle, self.cost_scale_)
        quality = np.where(bundle.availability, bundle.correctness, 0.0).astype(np.float32)
        return _CommTensors(
            query=torch.from_numpy(query).to(device),
            model_features=torch.from_numpy(model_features).to(device),
            availability=torch.from_numpy(bundle.availability.astype(bool)).to(device),
            quality=torch.from_numpy(quality).to(device),
            utility=torch.from_numpy(utility.astype(np.float32)).to(device),
            normalized_cost=torch.from_numpy(normalized_cost.astype(np.float32)).to(device),
        )

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
        self._train_loop(train_bundle, valid_bundle)

    def _loss(self, out: dict[str, Any], batch: _CommTensors) -> Any:
        availability = batch.availability
        cost_weight = _setting_cost_weight(self.setting)
        route_scores = out["mu_tilde"] - cost_weight * batch.normalized_cost
        route_scores = route_scores.masked_fill(~availability, -1e6)
        dist = _masked_gaussian_nll(out["mu"], out["sigma"], batch.quality, availability)
        pair = _masked_pairwise_loss(route_scores, batch.utility, availability)
        listwise = _masked_listwise_loss(route_scores, batch.utility, availability)
        util = _masked_regression_loss(route_scores, batch.utility, availability)
        mean = _masked_regression_loss(out["mu"], batch.quality, availability)
        residual = out["delta_cap"][availability].pow(2).mean() if availability.any() else route_scores.new_tensor(0.0)
        return (
            self.dist_weight * dist
            + self.pairwise_weight * pair
            + self.listwise_weight * listwise
            + self.utility_weight * util
            + self.mean_weight * mean
            + self.residual_weight * residual
        )

    def _validation_score(self, valid_bundle: RouterDatasetBundle) -> float:
        scores = self.predict_utilities(valid_bundle)
        target = _utility_targets(valid_bundle, self.setting, self.cost_scale_)
        selected = np.where(valid_bundle.availability, scores, -1e6).argmax(axis=1)
        rows = np.arange(len(selected))
        return float(np.nanmean(target[rows, selected]))

    def _train_loop(self, train_bundle: RouterDatasetBundle, valid_bundle: RouterDatasetBundle | None) -> None:
        if self.net_ is None:
            raise RuntimeError("Network was not initialized.")
        rng = np.random.default_rng(self.random_seed)
        train = self._bundle_tensors(train_bundle)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_state: dict[str, Any] | None = None
        best_valid = -1e18
        stale = 0
        losses: list[float] = []
        start_time = time.perf_counter()
        n = len(train_bundle.sample_ids)
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            epoch_losses: list[float] = []
            self.net_.train()
            for batch_start in range(0, n, self.batch_size):
                idx_np = order[batch_start : batch_start + self.batch_size]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=train.query.device)
                batch = _CommTensors(
                    query=train.query[idx],
                    model_features=train.model_features[idx],
                    availability=train.availability[idx],
                    quality=train.quality[idx],
                    utility=train.utility[idx],
                    normalized_cost=train.normalized_cost[idx],
                )
                out = self.net_(batch.query, batch.model_features, batch.availability)
                loss = self._loss(out, batch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                opt.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
            if valid_bundle is not None:
                self.net_.eval()
                with torch.no_grad():
                    score = self._validation_score(valid_bundle)
                if score > best_valid + 1e-9:
                    best_valid = score
                    best_state = copy.deepcopy({key: value.detach().cpu().clone() for key, value in self.net_.state_dict().items()})
                    stale = 0
                else:
                    stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.fit_summary_ = {
            "router_name": self.__class__.__name__,
            "paper_section": "3.2",
            "communication": "model_conditioned_capsule_read_pairwise_bias_feedback",
            "setting": self.setting,
            "epochs_requested": self.epochs,
            "epochs_completed": len(losses),
            "best_validation_selected_utility": best_valid if np.isfinite(best_valid) else None,
            "final_train_loss": losses[-1] if losses else None,
            "training_time_seconds": float(time.perf_counter() - start_time),
            "device": self.device_,
            "hidden_dim": self.hidden_dim,
            "num_query_tokens": self.num_query_tokens,
            "num_capsules": self.num_capsules,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "residual_bound": self.residual_bound,
            "feedback_temperature": self.feedback_temperature,
            "profile_summary": self.profile_summary_,
            "no_ecvl_supervision": True,
        }

    def predict_utilities(self, feature_bundle: RouterDatasetBundle, lambda_value: float | None = None) -> np.ndarray:
        del lambda_value
        if self.net_ is None:
            raise RuntimeError("Router must be fit before prediction.")
        tensors = self._bundle_tensors(feature_bundle)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(tensors.query, tensors.model_features, tensors.availability)
        arr = out["mu_tilde"].detach().cpu().numpy().astype(np.float32)
        arr[~feature_bundle.availability] = -1e6
        return arr

    def predict_distribution(self, feature_bundle: RouterDatasetBundle) -> dict[str, np.ndarray]:
        if self.net_ is None:
            raise RuntimeError("Router must be fit before prediction.")
        tensors = self._bundle_tensors(feature_bundle)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(tensors.query, tensors.model_features, tensors.availability)
        return {
            "mu": out["mu"].detach().cpu().numpy().astype(np.float32),
            "sigma": out["sigma"].detach().cpu().numpy().astype(np.float32),
            "delta_cap": out["delta_cap"].detach().cpu().numpy().astype(np.float32),
            "mu_tilde": out["mu_tilde"].detach().cpu().numpy().astype(np.float32),
        }
