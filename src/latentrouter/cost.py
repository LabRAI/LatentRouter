from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from latentrouter.data import RoutingData


class CostEstimator:
    """Per-query cost estimate c_hat_i(x) from per-model ridge regressions on the query features.

    When token prices are given, the regression predicts the number of generated tokens T_hat_i(x) and
    c_hat_i(x) = input_cost_i(x) + output_price_i * T_hat_i(x); otherwise it predicts the cost directly.
    Normalized estimates divide by the largest average training cost in the pool.
    """

    def __init__(self, alpha: float = 1e4) -> None:
        self.alpha = float(alpha)

    def fit(self, data: RoutingData) -> "CostEstimator":
        self.token_based_ = data.output_tokens is not None and data.output_price is not None
        if self.token_based_:
            target, self.price_ = data.output_tokens, np.asarray(data.output_price, dtype=float)
        else:
            target, self.price_ = data.cost.astype(float), np.ones(len(data.model_ids))
        self.models_ = []
        for i in range(target.shape[1]):
            ok = np.isfinite(target[:, i])
            self.models_.append(Ridge(alpha=self.alpha, random_state=42).fit(data.features[ok], target[ok, i]))
        mean_cost = np.nanmean(np.where(data.availability, data.cost, np.nan).astype(float), axis=0)
        self.scale_ = float(mean_cost.max())
        return self

    def predict(self, data: RoutingData) -> np.ndarray:
        cost = np.stack([np.maximum(0.0, m.predict(data.features)) for m in self.models_], axis=1) * self.price_[None, :]
        if self.token_based_:
            cost = cost + data.input_cost
        return cost

    def predict_normalized(self, data: RoutingData) -> np.ndarray:
        return self.predict(data) / self.scale_
