from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latentrouter.io import ensure_dir, read_json, stable_hash, write_json
from latentrouter.schemas import RouterDatasetBundle

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    TORCH_AVAILABLE = False


DEFAULT_CATEGORY_COLUMNS = ("group_id", "dataset_name", "mode_id")
DEFAULT_FAMILY_KEYS = (
    "claude",
    "gemini",
    "gpt",
    "qwen",
    "internvl",
    "janus",
    "gemma",
    "pixtral",
    "deepseek",
    "phi",
    "mistral",
    "flash",
    "pro",
    "sonnet",
    "vision",
    "vl",
)

PROFILE_ENCODER_TYPES = {"profile_mlp", "profile_mlp_basic", "profile_mlp_pairwise", "learned_id"}


def _is_profile_encoder(encoder_type: str) -> bool:
    return str(encoder_type) in {"profile_mlp", "profile_mlp_basic", "profile_mlp_pairwise"}


def _assert_train_only(bundle: RouterDatasetBundle) -> None:
    if "split" not in bundle.sample_frame.columns:
        return
    splits = set(bundle.sample_frame["split"].fillna("train").astype(str).str.lower().tolist())
    if not splits.issubset({"train", "calibration", "router_train"}):
        raise ValueError(f"Calibration memory can only be built from train rows, got splits={sorted(splits)!r}.")


def _parse_size_b(model_id: str) -> float:
    lowered = model_id.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", lowered)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*m\b", lowered)
    if match:
        return float(match.group(1)) / 1000.0
    return 0.0


def _family_flags(model_id: str) -> dict[str, float]:
    lowered = model_id.lower()
    return {f"family_{key}": float(key in lowered) for key in DEFAULT_FAMILY_KEYS}


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _select_rows_by_ids(bundle: RouterDatasetBundle, sample_ids: list[str] | np.ndarray) -> np.ndarray:
    requested = {str(sample_id) for sample_id in sample_ids}
    return np.asarray([str(sample_id) in requested for sample_id in bundle.sample_ids], dtype=bool)


def build_calibration_ids(
    train_bundle: RouterDatasetBundle,
    *,
    fraction: float = 0.2,
    min_samples: int = 16,
    max_samples: int | None = None,
    seed: int = 20260308,
) -> list[str]:
    _assert_train_only(train_bundle)
    sample_ids = np.asarray(train_bundle.sample_ids, dtype=str)
    if sample_ids.size == 0:
        return []
    target = max(int(math.ceil(sample_ids.size * float(fraction))), int(min_samples))
    target = min(target, sample_ids.size)
    if max_samples is not None:
        target = min(target, int(max_samples))
    rng = np.random.default_rng(seed)
    order = rng.permutation(sample_ids.size)
    selected = np.sort(sample_ids[order[:target]].astype(str))
    return selected.tolist()


def _category_values(frame: pd.DataFrame, category_columns: tuple[str, ...]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for column in category_columns:
        if column not in frame.columns:
            continue
        series = frame[column].fillna("unknown").astype(str)
        uniques = sorted(value for value in series.unique().tolist() if value)
        if len(uniques) > 1:
            values[column] = uniques
    return values


def _model_profile_vector(
    *,
    model_id: str,
    model_index: int,
    bundle: RouterDatasetBundle,
    row_mask: np.ndarray,
    category_values: dict[str, list[str]],
    cost_scale: float,
    latency_scale: float,
    size_scale: float,
) -> tuple[list[str], list[float], dict[str, Any]]:
    availability = bundle.availability[row_mask, model_index].astype(bool, copy=False)
    quality = bundle.correctness[row_mask, model_index]
    costs = bundle.costs[row_mask, model_index]
    token_counts = (
        bundle.token_counts[row_mask, model_index]
        if bundle.token_counts is not None
        else np.full(row_mask.sum(), np.nan, dtype=float)
    )
    overall_accuracy = _safe_mean(quality[availability])
    mean_cost = _safe_mean(costs[availability])
    mean_latency = _safe_mean(token_counts[availability])
    availability_rate = float(availability.mean()) if availability.size else 0.0
    size_b = _parse_size_b(model_id)

    feature_names = [
        "overall_accuracy",
        "normalized_cost",
        "normalized_latency",
        "availability_rate",
        "normalized_size_b",
        "log_size_b",
    ]
    feature_values = [
        overall_accuracy,
        mean_cost / max(cost_scale, 1e-12),
        mean_latency / max(latency_scale, 1e-12),
        availability_rate,
        size_b / max(size_scale, 1e-12),
        math.log1p(size_b),
    ]

    category_accuracy: dict[str, dict[str, float]] = {}
    frame = bundle.sample_frame.loc[row_mask].reset_index(drop=True)
    for column, values in category_values.items():
        series = frame[column].fillna("unknown").astype(str)
        category_accuracy[column] = {}
        for value in values:
            local = (series.to_numpy() == value) & availability
            score = _safe_mean(quality[local]) if np.any(local) else overall_accuracy
            category_accuracy[column][value] = score
            feature_names.append(f"{column}={value}:accuracy")
            feature_values.append(score)

    flags = _family_flags(model_id)
    for key in sorted(flags):
        feature_names.append(key)
        feature_values.append(flags[key])

    profile = {
        "model_id": model_id,
        "overall_accuracy": overall_accuracy,
        "category_accuracy": category_accuracy,
        "mean_cost": mean_cost,
        "mean_latency": mean_latency,
        "availability_rate": availability_rate,
        "size_b": size_b,
        "family_flags": flags,
        "feature_names": feature_names,
        "feature_vector": feature_values,
    }
    return feature_names, feature_values, profile


def _pairwise_calibration_features(
    *,
    bundle: RouterDatasetBundle,
    row_mask: np.ndarray,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    quality = bundle.correctness[row_mask]
    availability = bundle.availability[row_mask].astype(bool, copy=False)
    model_ids = list(bundle.model_ids)
    by_model: dict[str, dict[str, dict[str, float]]] = {model_id: {} for model_id in model_ids}
    rows: list[dict[str, Any]] = []
    for left_idx, left_model in enumerate(model_ids):
        for right_idx, right_model in enumerate(model_ids):
            if left_idx == right_idx:
                stats = {
                    "valid_count": float(int(availability[:, left_idx].sum())) if availability.size else 0.0,
                    "win_rate": 0.0,
                    "tie_rate": 1.0,
                    "loss_rate": 0.0,
                }
            else:
                valid = (
                    availability[:, left_idx]
                    & availability[:, right_idx]
                    & np.isfinite(quality[:, left_idx])
                    & np.isfinite(quality[:, right_idx])
                )
                if np.any(valid):
                    left_quality = quality[valid, left_idx]
                    right_quality = quality[valid, right_idx]
                    wins = left_quality > right_quality
                    ties = left_quality == right_quality
                    losses = left_quality < right_quality
                    denom = float(valid.sum())
                    stats = {
                        "valid_count": float(int(valid.sum())),
                        "win_rate": float(wins.sum() / denom),
                        "tie_rate": float(ties.sum() / denom),
                        "loss_rate": float(losses.sum() / denom),
                    }
                else:
                    stats = {"valid_count": 0.0, "win_rate": 0.0, "tie_rate": 0.0, "loss_rate": 0.0}
            by_model[left_model][right_model] = stats
            rows.append(
                {
                    "model_id": left_model,
                    "opponent_model_id": right_model,
                    "valid_count": int(stats["valid_count"]),
                    "win_rate": float(stats["win_rate"]),
                    "tie_rate": float(stats["tie_rate"]),
                    "loss_rate": float(stats["loss_rate"]),
                }
            )
    return by_model, rows


def _failure_correlation_features(
    *,
    bundle: RouterDatasetBundle,
    row_mask: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    quality = bundle.correctness[row_mask]
    availability = bundle.availability[row_mask].astype(bool, copy=False)
    model_ids = list(bundle.model_ids)
    by_model: dict[str, dict[str, float]] = {model_id: {} for model_id in model_ids}
    rows: list[dict[str, Any]] = []
    for left_idx, left_model in enumerate(model_ids):
        for right_idx, right_model in enumerate(model_ids):
            valid = (
                availability[:, left_idx]
                & availability[:, right_idx]
                & np.isfinite(quality[:, left_idx])
                & np.isfinite(quality[:, right_idx])
            )
            if left_idx == right_idx:
                corr = 1.0
                joint_failure = float(np.mean(quality[valid, left_idx] <= threshold)) if np.any(valid) else 0.0
            elif np.any(valid):
                left_failure = (quality[valid, left_idx] <= threshold).astype(float)
                right_failure = (quality[valid, right_idx] <= threshold).astype(float)
                joint_failure = float(np.mean((left_failure > 0.5) & (right_failure > 0.5)))
                if np.std(left_failure) <= 1e-8 or np.std(right_failure) <= 1e-8:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(left_failure, right_failure)[0, 1])
            else:
                corr = 0.0
                joint_failure = 0.0
            if not np.isfinite(corr):
                corr = 0.0
            by_model[left_model][right_model] = corr
            rows.append(
                {
                    "model_id": left_model,
                    "opponent_model_id": right_model,
                    "failure_correlation": float(corr),
                    "joint_failure_rate": float(joint_failure),
                }
            )
    return by_model, rows


def build_memory_from_bundle(
    train_bundle: RouterDatasetBundle,
    *,
    calibration_ids: list[str] | np.ndarray | None = None,
    calibration_fraction: float = 0.2,
    min_calibration_samples: int = 16,
    max_calibration_samples: int | None = None,
    seed: int = 20260308,
    encoder_type: str = "profile_mlp",
    d_model: int = 128,
    category_columns: tuple[str, ...] = DEFAULT_CATEGORY_COLUMNS,
) -> "ModelCapabilityMemory":
    _assert_train_only(train_bundle)
    if encoder_type not in PROFILE_ENCODER_TYPES:
        raise ValueError(f"encoder_type must be one of {sorted(PROFILE_ENCODER_TYPES)!r}.")
    selected_ids = (
        [str(sample_id) for sample_id in calibration_ids]
        if calibration_ids is not None
        else build_calibration_ids(
            train_bundle,
            fraction=calibration_fraction,
            min_samples=min_calibration_samples,
            max_samples=max_calibration_samples,
            seed=seed,
        )
    )
    row_mask = _select_rows_by_ids(train_bundle, selected_ids)
    if not np.any(row_mask):
        raise ValueError("No calibration rows matched the requested calibration IDs.")

    calib_costs = train_bundle.costs[row_mask]
    calib_avail = train_bundle.availability[row_mask]
    cost_scale = float(np.nanmax(calib_costs[calib_avail])) if np.any(calib_avail) else 1.0
    if train_bundle.token_counts is not None and np.any(calib_avail):
        latency_scale = float(np.nanmax(train_bundle.token_counts[row_mask][calib_avail]))
    else:
        latency_scale = 1.0
    size_scale = max((_parse_size_b(model_id) for model_id in train_bundle.model_ids), default=1.0)
    category_values = _category_values(train_bundle.sample_frame.loc[row_mask].reset_index(drop=True), category_columns)
    pairwise_by_model: dict[str, dict[str, dict[str, float]]] = {}
    pairwise_rows: list[dict[str, Any]] = []
    failure_corr_by_model: dict[str, dict[str, float]] = {}
    failure_corr_rows: list[dict[str, Any]] = []
    if encoder_type == "profile_mlp_pairwise":
        pairwise_by_model, pairwise_rows = _pairwise_calibration_features(bundle=train_bundle, row_mask=row_mask)
        failure_corr_by_model, failure_corr_rows = _failure_correlation_features(bundle=train_bundle, row_mask=row_mask)

    profiles: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    for model_index, model_id in enumerate(train_bundle.model_ids):
        names, _values, profile = _model_profile_vector(
            model_id=model_id,
            model_index=model_index,
            bundle=train_bundle,
            row_mask=row_mask,
            category_values=category_values,
            cost_scale=max(cost_scale, 1e-12),
            latency_scale=max(latency_scale, 1e-12),
            size_scale=max(size_scale, 1e-12),
        )
        if encoder_type == "profile_mlp_pairwise":
            for opponent_model_id in train_bundle.model_ids:
                stats = pairwise_by_model[model_id][opponent_model_id]
                for key in ("win_rate", "tie_rate", "loss_rate"):
                    names.append(f"pairwise_vs={opponent_model_id}:{key}")
                    profile["feature_vector"].append(float(stats[key]))
                names.append(f"failure_corr_vs={opponent_model_id}")
                profile["feature_vector"].append(float(failure_corr_by_model[model_id][opponent_model_id]))
            profile["pairwise_calibration"] = pairwise_by_model[model_id]
            profile["failure_correlation"] = failure_corr_by_model[model_id]
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Model profile feature names are not aligned across models.")
        profiles.append(profile)

    return ModelCapabilityMemory(
        profiles=profiles,
        calibration_ids=selected_ids,
        feature_names=feature_names or [],
        d_model=d_model,
        encoder_type=encoder_type,
        seed=seed,
        pairwise_calibration_stats=pairwise_rows,
        failure_correlation_stats=failure_corr_rows,
    )


class ModelCapabilityMemory(nn.Module if TORCH_AVAILABLE else object):
    def __init__(
        self,
        profiles: list[dict[str, Any]],
        *,
        calibration_ids: list[str] | None = None,
        feature_names: list[str] | None = None,
        d_model: int = 128,
        encoder_type: str = "profile_mlp",
        seed: int = 20260308,
        pairwise_calibration_stats: list[dict[str, Any]] | None = None,
        failure_correlation_stats: list[dict[str, Any]] | None = None,
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        if encoder_type not in PROFILE_ENCODER_TYPES:
            raise ValueError(f"encoder_type must be one of {sorted(PROFILE_ENCODER_TYPES)!r}.")
        self.encoder_type = str(encoder_type)
        self.d_model = int(d_model)
        self.seed = int(seed)
        self.calibration_ids = [str(sample_id) for sample_id in (calibration_ids or [])]
        self.pairwise_calibration_stats = [dict(row) for row in (pairwise_calibration_stats or [])]
        self.failure_correlation_stats = [dict(row) for row in (failure_correlation_stats or [])]
        self.profiles: dict[str, dict[str, Any]] = {str(profile["model_id"]): dict(profile) for profile in profiles}
        self.model_ids = list(self.profiles.keys())
        self.feature_names = list(feature_names or (profiles[0].get("feature_names", []) if profiles else []))
        self._refresh_matrices()

    def _refresh_matrices(self) -> None:
        rows: list[np.ndarray] = []
        for model_id in self.model_ids:
            profile = self.profiles[model_id]
            values = np.asarray(profile.get("feature_vector", []), dtype=np.float32)
            if self.feature_names and values.shape[0] != len(self.feature_names):
                raise ValueError(
                    f"Profile vector for {model_id!r} has length {values.shape[0]}, expected {len(self.feature_names)}."
                )
            rows.append(values)
        self.profile_matrix = np.stack(rows, axis=0).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
        self.cost_vector = np.asarray(
            [float(self.profiles[model_id].get("mean_cost", 0.0)) for model_id in self.model_ids],
            dtype=np.float32,
        )
        self.latency_vector = np.asarray(
            [float(self.profiles[model_id].get("mean_latency", 0.0)) for model_id in self.model_ids],
            dtype=np.float32,
        )

    @property
    def profile_dim(self) -> int:
        return int(self.profile_matrix.shape[1]) if self.profile_matrix.ndim == 2 else 0

    def get_profile_matrix(self, model_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        costs: list[float] = []
        latencies: list[float] = []
        mask: list[bool] = []
        zero = np.zeros(self.profile_dim, dtype=np.float32)
        lookup = {model_id: idx for idx, model_id in enumerate(self.model_ids)}
        for model_name in model_names:
            idx = lookup.get(str(model_name))
            if idx is None:
                rows.append(zero.copy())
                costs.append(0.0)
                latencies.append(0.0)
                mask.append(False)
            else:
                rows.append(self.profile_matrix[idx])
                costs.append(float(self.cost_vector[idx]))
                latencies.append(float(self.latency_vector[idx]))
                mask.append(True)
        return (
            np.stack(rows, axis=0).astype(np.float32) if rows else np.zeros((0, self.profile_dim), dtype=np.float32),
            np.asarray(costs, dtype=np.float32),
            np.asarray(latencies, dtype=np.float32),
            np.asarray(mask, dtype=bool),
        )

    def get_memory_tokens(self, model_names: list[str]):
        matrix, costs, latencies, mask = self.get_profile_matrix(model_names)
        if not TORCH_AVAILABLE:
            return matrix, costs, latencies, mask
        return (
            torch.from_numpy(matrix),
            torch.from_numpy(costs),
            torch.from_numpy(latencies),
            torch.from_numpy(mask),
        )

    def pairwise_calibration_stats_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.pairwise_calibration_stats,
            columns=["model_id", "opponent_model_id", "valid_count", "win_rate", "tie_rate", "loss_rate"],
        )

    def failure_correlation_stats_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.failure_correlation_stats,
            columns=["model_id", "opponent_model_id", "failure_correlation", "joint_failure_rate"],
        )

    def add_model(self, model_name: str, profile: dict[str, Any]) -> None:
        profile = dict(profile)
        profile["model_id"] = str(model_name)
        if "feature_vector" not in profile:
            raise ValueError("Added model profile must contain a feature_vector.")
        if self.feature_names and len(profile["feature_vector"]) != len(self.feature_names):
            raise ValueError("Added model profile feature vector does not match existing memory feature dimension.")
        self.profiles[str(model_name)] = profile
        if str(model_name) not in self.model_ids:
            self.model_ids.append(str(model_name))
        self._refresh_matrices()

    def remove_model(self, model_name: str) -> None:
        self.profiles.pop(str(model_name), None)
        self.model_ids = [model_id for model_id in self.model_ids if model_id != str(model_name)]
        self._refresh_matrices()

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder_type": self.encoder_type,
            "d_model": self.d_model,
            "seed": self.seed,
            "calibration_ids": self.calibration_ids,
            "feature_names": self.feature_names,
            "model_ids": self.model_ids,
            "profiles": [self.profiles[model_id] for model_id in self.model_ids],
            "pairwise_calibration_stats": self.pairwise_calibration_stats,
            "failure_correlation_stats": self.failure_correlation_stats,
            "profile_dim": self.profile_dim,
            "integrity_hash": stable_hash(json.dumps(self.model_ids + self.feature_names, sort_keys=True)),
        }

    def save(self, path: str | Path, *, pt_path: str | Path | None = None) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        if path.suffix == ".pt":
            if not TORCH_AVAILABLE:
                raise RuntimeError("Torch is required to save .pt memory artifacts.")
            torch.save(self.to_dict(), path)
        else:
            write_json(path, self.to_dict())
        if pt_path is not None:
            if not TORCH_AVAILABLE:
                raise RuntimeError("Torch is required to save .pt memory artifacts.")
            pt_path = Path(pt_path)
            ensure_dir(pt_path.parent)
            torch.save(self.to_dict(), pt_path)

    @classmethod
    def load(cls, path: str | Path) -> "ModelCapabilityMemory":
        path = Path(path)
        if path.suffix == ".pt":
            if not TORCH_AVAILABLE:
                raise RuntimeError("Torch is required to load .pt memory artifacts.")
            payload = torch.load(path, map_location="cpu")
        else:
            payload = read_json(path)
        return cls(
            profiles=list(payload.get("profiles", [])),
            calibration_ids=list(payload.get("calibration_ids", [])),
            feature_names=list(payload.get("feature_names", [])),
            d_model=int(payload.get("d_model", 128)),
            encoder_type=str(payload.get("encoder_type", "profile_mlp")),
            seed=int(payload.get("seed", 20260308)),
            pairwise_calibration_stats=list(payload.get("pairwise_calibration_stats", [])),
            failure_correlation_stats=list(payload.get("failure_correlation_stats", [])),
        )
