from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from latentrouter.config import EvaluationConfig
from latentrouter.schemas import NormalizedArtifacts, RouterDatasetBundle


class BenchmarkAdapter(ABC):
    benchmark_id: str
    protocol_id: str

    @abstractmethod
    def prepare(
        self,
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
        raise NotImplementedError


class BenchmarkProtocol(ABC):
    benchmark_id: str
    protocol_id: str

    @abstractmethod
    def evaluate(
        self,
        router_name: str,
        predicted_utilities: np.ndarray,
        bundle: RouterDatasetBundle,
        config: EvaluationConfig,
        run_dir: str | Path,
        prediction_wall_time_seconds: float | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
