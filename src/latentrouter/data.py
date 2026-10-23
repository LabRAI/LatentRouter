from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass
class RoutingData:
    """Outcome traces of a candidate model pool on a set of queries.

    features:      (n, d) query features
    quality:       (n, K) normalized answer quality in [0, 1]
    cost:          (n, K) realized cost of answering each query with each model
    availability:  (n, K) boolean mask of available models
    slices:        per-query slice labels used by the calibration profiles
    latency:       (n, K) latency of each model on each query (optional)
    input_cost:    (n, K) cost of the prompt, known before generation (optional)
    output_tokens: (n, K) number of generated tokens (optional)
    output_price:  (K,) price per generated token (optional)
    """

    sample_ids: np.ndarray
    model_ids: list[str]
    features: np.ndarray
    quality: np.ndarray | None = None
    cost: np.ndarray | None = None
    availability: np.ndarray | None = None
    slices: pd.DataFrame = field(default_factory=pd.DataFrame)
    latency: np.ndarray | None = None
    input_cost: np.ndarray | None = None
    output_tokens: np.ndarray | None = None
    output_price: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.sample_ids = np.asarray(self.sample_ids).astype(str)
        self.model_ids = list(self.model_ids)
        if self.availability is None and self.quality is not None:
            self.availability = np.ones_like(self.quality, dtype=bool)
        elif self.availability is None:
            self.availability = np.ones((len(self.features), len(self.model_ids)), dtype=bool)
        self.availability = np.asarray(self.availability, dtype=bool)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def subset(self, rows: np.ndarray) -> "RoutingData":
        def take(a):
            return None if a is None else a[rows]

        return replace(
            self,
            sample_ids=self.sample_ids[rows],
            features=self.features[rows],
            quality=take(self.quality),
            cost=take(self.cost),
            availability=self.availability[rows],
            slices=self.slices.iloc[rows].reset_index(drop=True) if len(self.slices.columns) else pd.DataFrame(),
            latency=take(self.latency),
            input_cost=take(self.input_cost),
            output_tokens=take(self.output_tokens),
        )
