from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

FAMILY_KEYS = (
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


def _size_in_billions(model_id: str) -> float:
    name = model_id.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*m\b", name)
    if match:
        return float(match.group(1)) / 1000.0
    return 0.0


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def calibration_rows(n: int, *, fraction: float, max_samples: int | None, seed: int, min_samples: int = 16) -> np.ndarray:
    target = min(max(int(math.ceil(n * fraction)), min_samples), n)
    if max_samples is not None:
        target = min(target, int(max_samples))
    order = np.random.default_rng(seed).permutation(n)
    rows = np.zeros(n, dtype=bool)
    rows[order[:target]] = True
    return rows


def calibration_profiles(
    model_ids: list[str],
    quality: np.ndarray,
    cost: np.ndarray,
    availability: np.ndarray,
    slices: pd.DataFrame,
    rows: np.ndarray,
    latency: np.ndarray | None = None,
) -> np.ndarray:
    """Calibration profile p_i of every model, computed on the calibration rows of the training split.

    Each profile holds the model's overall accuracy, normalized cost and latency, size and family metadata,
    its accuracy on every benchmark slice, and its win/tie/loss rates and failure correlation against every
    other candidate.
    """
    q, c, avail = quality[rows], cost[rows], availability[rows]
    lat = latency[rows] if latency is not None else np.full(q.shape, np.nan)
    cost_scale = max(float(np.nanmax(c[avail])) if avail.any() else 1.0, 1e-12)
    latency_scale = max(float(np.nanmax(lat[avail])) if latency is not None and avail.any() else 1.0, 1e-12)
    size_scale = max(max((_size_in_billions(m) for m in model_ids), default=1.0), 1e-12)
    frame = slices.loc[rows].reset_index(drop=True) if len(slices.columns) else pd.DataFrame()
    slice_values = {}
    for column in frame.columns:
        values = sorted(v for v in frame[column].fillna("unknown").astype(str).unique().tolist() if v)
        if len(values) > 1:
            slice_values[column] = values

    finite = np.isfinite(q)
    failure = q <= 0.5
    profiles = []
    for i, model_id in enumerate(model_ids):
        a = avail[:, i]
        accuracy = _mean(q[a, i])
        size = _size_in_billions(model_id)
        vector = [
            accuracy,
            _mean(c[a, i]) / cost_scale,
            _mean(lat[a, i]) / latency_scale,
            float(a.mean()) if a.size else 0.0,
            size / size_scale,
            math.log1p(size),
        ]
        for column, values in slice_values.items():
            labels = frame[column].fillna("unknown").astype(str).to_numpy()
            for value in values:
                local = (labels == value) & a
                vector.append(_mean(q[local, i]) if local.any() else accuracy)
        name = model_id.lower()
        vector.extend(float(key in name) for key in sorted(FAMILY_KEYS))
        for j in range(len(model_ids)):
            valid = a & avail[:, j] & finite[:, i] & finite[:, j]
            if i == j:
                vector.extend([0.0, 1.0, 0.0, 1.0])
                continue
            if valid.any():
                left, right = q[valid, i], q[valid, j]
                n = float(valid.sum())
                vector.extend([(left > right).sum() / n, (left == right).sum() / n, (left < right).sum() / n])
                fl, fr = failure[valid, i].astype(float), failure[valid, j].astype(float)
                corr = 0.0 if fl.std() <= 1e-8 or fr.std() <= 1e-8 else float(np.corrcoef(fl, fr)[0, 1])
                vector.append(corr if np.isfinite(corr) else 0.0)
            else:
                vector.extend([0.0, 0.0, 0.0, 0.0])
        profiles.append(vector)
    return np.asarray(profiles, dtype=np.float32)
