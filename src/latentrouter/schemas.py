from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class NormalizedArtifacts:
    processed_dir: Path
    samples_path: Path
    outcomes_path: Path
    models_path: Path
    manifest_path: Path
    splits_path: Path


@dataclass(slots=True)
class FeatureArtifacts:
    feature_dir: Path
    sample_index_path: Path
    text_embeddings_path: Path
    image_embeddings_path: Path
    side_features_path: Path
    fused_features_path: Path
    manifest_path: Path


@dataclass(slots=True)
class RouterDatasetBundle:
    sample_ids: np.ndarray
    model_ids: list[str]
    features: np.ndarray
    correctness: np.ndarray
    costs: np.ndarray
    token_counts: np.ndarray | None
    availability: np.ndarray
    sample_frame: pd.DataFrame
    benchmark_id: str = "mmr"
    protocol_id: str = "mmr_cost_quality"
    evaluation_metadata: dict[str, Any] | None = None

    def slice(self, mask: np.ndarray) -> "RouterDatasetBundle":
        return RouterDatasetBundle(
            sample_ids=self.sample_ids[mask],
            model_ids=list(self.model_ids),
            features=self.features[mask],
            correctness=self.correctness[mask],
            costs=self.costs[mask],
            token_counts=None if self.token_counts is None else self.token_counts[mask],
            availability=self.availability[mask],
            sample_frame=self.sample_frame.loc[mask].reset_index(drop=True),
            benchmark_id=self.benchmark_id,
            protocol_id=self.protocol_id,
            evaluation_metadata=dict(self.evaluation_metadata or {}),
        )
