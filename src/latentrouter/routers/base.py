from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from latentrouter.schemas import RouterDatasetBundle


class BaseRouter(ABC):
    supports_lambda_conditioning = False

    def __init__(self, model_ids: list[str], random_seed: int = 20260308, **_: object):
        self.model_ids = list(model_ids)
        self.random_seed = random_seed

    @abstractmethod
    def fit(self, train_bundle: RouterDatasetBundle, valid_bundle: RouterDatasetBundle | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_utilities(self, feature_bundle: RouterDatasetBundle, lambda_value: float | None = None) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "BaseRouter":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, BaseRouter):
            raise TypeError(f"Serialized object is not a BaseRouter: {type(obj)!r}")
        return obj
