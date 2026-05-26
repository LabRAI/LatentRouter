from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    benchmark: str = "mmr"
    source: str = "hf://gh0stHunter/MMR-Bench"
    source_split: str = "train"
    raw_dir: str = "data/raw/mmr_bench"
    processed_dir: str = "data/processed/mmr_bench"
    split_manifest: str | None = None
    train_fraction: float = 0.7
    validation_fraction: float = 0.1
    test_fraction: float = 0.2
    seed: int = 20260308
    adapter_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeaturesConfig:
    backend: str = "openclip"
    openclip_model: str = "ViT-B-32"
    openclip_pretrained: str = "laion2b_s34b_b79k"
    hashing_dim: int = 256
    coconut_repo_path: str | None = None
    coconut_model_id: str = "openai-community/gpt2"
    coconut_checkpoint: str | None = None
    coconut_checkpoint_filename: str = "pytorch_model.bin"
    coconut_latent_tokens: int = 2
    coconut_max_length: int = 256
    batch_size: int = 32
    device: str = "cpu"
    output_dir: str | None = None
    force: bool = False


@dataclass(slots=True)
class RouterConfig:
    name: str = "knn"
    model_path: str | None = None
    random_seed: int = 20260308
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationConfig:
    lambda_min_exp: float = -4.0
    lambda_max_exp: float = 4.0
    num_lambdas: int = 201
    output_dir: str = "artifacts/runs"
    aggregate_by: list[str] = field(default_factory=lambda: ["dataset_name", "mode_id"])
    protocol_options: dict[str, Any] = field(default_factory=dict)
    measure_throughput: bool = False


@dataclass(slots=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge_defaults(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_run_config(path: str | Path | None = None) -> RunConfig:
    base = RunConfig()
    if path is None:
        return base

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    merged = _merge_defaults(
        {
            "data": asdict(base.data),
            "features": asdict(base.features),
            "router": asdict(base.router),
            "evaluation": asdict(base.evaluation),
        },
        data,
    )
    return RunConfig(
        data=DataConfig(**merged["data"]),
        features=FeaturesConfig(**merged["features"]),
        router=RouterConfig(**merged["router"]),
        evaluation=EvaluationConfig(**merged["evaluation"]),
    )
