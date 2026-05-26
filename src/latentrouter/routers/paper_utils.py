from __future__ import annotations

import numpy as np

from latentrouter.memory.model_capability_memory import build_memory_from_bundle
from latentrouter.schemas import RouterDatasetBundle

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - runtime guard.
    torch = None
    F = None
    TORCH_AVAILABLE = False


def _device_name(requested: str) -> str:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for LatentRouter.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def _setting_cost_weight(setting: str) -> float:
    return 0.03 if str(setting) == "performance_cost" else 0.0


def _normalized_cost(bundle: RouterDatasetBundle, scale: float | None = None) -> np.ndarray:
    availability = bundle.availability.astype(bool, copy=False)
    if scale is None:
        scale = float(np.nanmax(bundle.costs[availability])) if np.any(availability) else 1.0
    out = np.zeros_like(bundle.costs, dtype=np.float32)
    out[availability] = (bundle.costs[availability] / max(float(scale), 1e-12)).astype(np.float32)
    return out


def _utility_targets(bundle: RouterDatasetBundle, setting: str, cost_scale: float | None = None) -> np.ndarray:
    availability = bundle.availability.astype(bool, copy=False)
    utility = np.full(bundle.correctness.shape, -1e6, dtype=np.float32)
    cost = _normalized_cost(bundle, cost_scale)
    weight = _setting_cost_weight(setting)
    utility[availability] = (bundle.correctness[availability] - weight * cost[availability]).astype(np.float32)
    return utility


def _profile_matrix(
    train_bundle: RouterDatasetBundle,
    model_ids: list[str],
    *,
    seed: int,
    encoder_type: str,
    calibration_fraction: float,
    max_calibration_samples: int | None,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    memory = build_memory_from_bundle(
        train_bundle,
        calibration_fraction=calibration_fraction,
        min_calibration_samples=16,
        max_calibration_samples=max_calibration_samples,
        seed=seed,
        encoder_type=encoder_type,
    )
    matrix, _costs, _latencies, mask = memory.get_profile_matrix(model_ids)
    if not bool(np.all(mask)):
        missing = [model for model, ok in zip(model_ids, mask) if not ok]
        raise ValueError(f"Capability profiles missing for model(s): {missing}")
    return matrix.astype(np.float32), list(memory.feature_names), memory.to_dict()


def _masked_gaussian_nll(mu, sigma, target, availability):
    mask = availability & torch.isfinite(target)
    if not mask.any():
        return torch.zeros((), device=mu.device)
    err = target[mask] - mu[mask]
    sig = sigma[mask].clamp_min(1e-4)
    return (0.5 * (err / sig).pow(2) + torch.log(sig)).mean()


def _masked_listwise_loss(scores, targets, availability, temperature: float = 0.05):
    masked_scores = scores.masked_fill(~availability, -1e6)
    masked_targets = targets.masked_fill(~availability, -1e6)
    target_probs = torch.softmax(masked_targets / max(float(temperature), 1e-6), dim=1)
    log_probs = torch.log_softmax(masked_scores, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1)
    valid_rows = availability.any(dim=1)
    return loss[valid_rows].mean() if valid_rows.any() else torch.zeros((), device=scores.device)


def _masked_pairwise_loss(scores, targets, availability):
    valid = availability & torch.isfinite(targets)
    pair_mask = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(scores.shape[1], dtype=torch.bool, device=scores.device)[None, :, :]
    pair_mask = pair_mask & ~eye
    if not pair_mask.any():
        return torch.zeros((), device=scores.device)
    pred_delta = scores[:, :, None] - scores[:, None, :]
    target_pref = (targets[:, :, None] > targets[:, None, :]).float()
    return F.binary_cross_entropy_with_logits(pred_delta[pair_mask], target_pref[pair_mask])


def _masked_regression_loss(scores, targets, availability):
    mask = availability & torch.isfinite(targets)
    if not mask.any():
        return torch.zeros((), device=scores.device)
    return F.mse_loss(scores[mask], targets[mask])
