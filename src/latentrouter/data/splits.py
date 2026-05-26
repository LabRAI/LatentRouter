from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def _load_manifest(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if target.suffix == ".parquet":
        frame = pd.read_parquet(target)
    elif target.suffix == ".json":
        frame = pd.read_json(target)
    else:
        frame = pd.read_csv(target)
    if not {"sample_id", "split"}.issubset(frame.columns):
        raise ValueError("Split manifest must contain 'sample_id' and 'split' columns.")
    return frame[["sample_id", "split"]].copy()


def _allocate_counts(size: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if size <= 0:
        return (0, 0, 0)
    if size == 1:
        return (1, 0, 0)
    if size == 2:
        return (1, 0, 1)
    if size == 3:
        return (1, 1, 1)

    raw = np.array(fractions, dtype=float) * size
    counts = np.floor(raw).astype(int)
    remainder = size - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for idx in order[:remainder]:
        counts[idx] += 1

    active = [idx for idx, fraction in enumerate(fractions) if fraction > 0]
    for idx in active:
        if counts[idx] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[idx] += 1

    train, val, test = counts.tolist()
    if train <= 0:
        train = 1
        if test > val:
            test -= 1
        else:
            val -= 1
    if train + val + test != size:
        test += size - (train + val + test)
    return (train, val, test)


def create_splits(
    samples: pd.DataFrame,
    split_manifest: str | Path | None = None,
    seed: int = 20260308,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    if split_manifest:
        manifest = _load_manifest(split_manifest)
        return samples[["sample_id"]].merge(manifest, on="sample_id", how="left")

    if not math.isclose(train_fraction + validation_fraction + test_fraction, 1.0, rel_tol=1e-6):
        raise ValueError("Split fractions must sum to 1.0.")

    fractions = (train_fraction, validation_fraction, test_fraction)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    grouping = group_columns or ["dataset_name", "mode_id"]
    grouped = samples.groupby(grouping, dropna=False, sort=True)
    for _, frame in grouped:
        indices = frame.index.to_numpy()
        rng.shuffle(indices)
        n_train, n_val, n_test = _allocate_counts(len(indices), fractions)
        split_names = (
            ["train"] * n_train +
            ["val"] * n_val +
            ["test"] * n_test
        )
        for idx, split_name in zip(indices, split_names, strict=True):
            rows.append({"sample_id": samples.at[idx, "sample_id"], "split": split_name})

    split_frame = pd.DataFrame(rows)
    return samples[["sample_id"]].merge(split_frame, on="sample_id", how="left")
