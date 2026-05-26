from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from latentrouter.benchmarks.mmr import normalize_source as normalize_mmr_source
from latentrouter.benchmarks.mmr import parse_dataset_fields as _parse_dataset_fields
from latentrouter.benchmarks.registry import create_adapter
from latentrouter.schemas import NormalizedArtifacts


def normalize_source(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    return normalize_mmr_source(frame)


def prepare_benchmark(
    benchmark: str,
    source: str,
    processed_dir: str | Path,
    source_split: str = "train",
    split_manifest: str | Path | None = None,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 20260308,
    adapter_options: dict[str, Any] | None = None,
) -> NormalizedArtifacts:
    adapter = create_adapter(benchmark)
    return adapter.prepare(
        source=source,
        processed_dir=processed_dir,
        source_split=source_split,
        split_manifest=split_manifest,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        adapter_options=adapter_options,
    )


def prepare_dataset(
    source: str,
    processed_dir: str | Path,
    source_split: str = "train",
    split_manifest: str | Path | None = None,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 20260308,
    benchmark: str = "mmr",
    adapter_options: dict[str, Any] | None = None,
) -> NormalizedArtifacts:
    return prepare_benchmark(
        benchmark=benchmark,
        source=source,
        processed_dir=processed_dir,
        source_split=source_split,
        split_manifest=split_manifest,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        adapter_options=adapter_options,
    )


__all__ = ["_parse_dataset_fields", "normalize_source", "prepare_benchmark", "prepare_dataset"]
