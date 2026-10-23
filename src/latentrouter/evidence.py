from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


class CalibrationEvidence:
    """Query-specific calibration statistics b_i(x).

    For a query x, the k most similar training queries (cosine distance on standardized features) form a
    query-specific slice. The accuracy of each model on this neighborhood is shrunk toward its overall training
    accuracy, pi_loc = (s + kappa * y_bar) / (n + kappa). A per-model ridge probe on the same features gives a
    linear estimate r_i(x). The calibrated estimate averages both in logit space,
    logit pi_i(x) = (logit pi_loc + logit r_i(x)) / 2, and

        b_i(x) = [y_hat_i(x), y_hat_i(x) - y_bar_i, r_i(x), pi_i(x)].

    For a training query, the neighborhood excludes the query itself and r_i(x) is a cross-fitted prediction.
    """

    def __init__(self, num_neighbors: int = 32, shrinkage: float = 5.0, probe_alpha: float = 1e4, probe_folds: int = 5,
                 seed: int = 0) -> None:
        self.num_neighbors = int(num_neighbors)
        self.shrinkage = float(shrinkage)
        self.probe_alpha = float(probe_alpha)
        self.probe_folds = int(probe_folds)
        self.seed = int(seed)

    def fit(self, features: np.ndarray, quality: np.ndarray, availability: np.ndarray) -> "CalibrationEvidence":
        self.train_features_ = features.astype(np.float32)
        self.train_quality_ = np.where(availability, quality, np.nan).astype(np.float32)
        self.overall_accuracy_ = np.nanmean(self.train_quality_, axis=0)
        self.index_ = NearestNeighbors(metric="cosine").fit(self.train_features_)
        y = self.train_quality_
        self.probes_ = []
        self.train_probe_ = np.zeros_like(y)
        for i in range(y.shape[1]):
            ok = np.isfinite(y[:, i])
            self.probes_.append(Ridge(alpha=self.probe_alpha).fit(features[ok], y[ok, i]))
            for fit_rows, held_out in KFold(self.probe_folds, shuffle=True, random_state=self.seed).split(features):
                m = ok[fit_rows]
                probe = Ridge(alpha=self.probe_alpha).fit(features[fit_rows][m], y[fit_rows][m, i])
                self.train_probe_[held_out, i] = probe.predict(features[held_out])
        return self

    def _neighborhood(self, features: np.ndarray, training: bool) -> np.ndarray:
        q = features.astype(np.float32)
        k = min(self.num_neighbors, len(self.train_features_) - 1)
        _, idx = self.index_.kneighbors(q, n_neighbors=k + 1)
        y, y_bar = self.train_quality_, self.overall_accuracy_
        out = np.zeros((len(q), y.shape[1], 3), dtype=np.float32)
        for r in range(len(q)):
            nb = idx[r][idx[r] != r][:k] if training else idx[r][:k]
            local = np.nanmean(y[nb], axis=0)
            count = np.sum(np.isfinite(y[nb]), axis=0)
            successes = np.nansum(y[nb], axis=0)
            out[r, :, 0] = local
            out[r, :, 1] = local - y_bar
            out[r, :, 2] = (successes + self.shrinkage * y_bar) / (count + self.shrinkage)
        return out

    def transform(self, features: np.ndarray, training: bool = False) -> np.ndarray:
        """Return b_i(x) with shape (n, K, 4). Use training=True only for the rows passed to fit, in order."""
        local = self._neighborhood(features, training)
        if training:
            probe = self.train_probe_
        else:
            probe = np.stack([p.predict(features) for p in self.probes_], axis=1)
        probe = np.clip(probe, 0.01, 0.99).astype(np.float32)
        pi_loc = np.clip(local[..., 2], 1e-3, 1 - 1e-3)
        pi = (1.0 / (1.0 + np.exp(-(_logit(pi_loc) + _logit(probe)) / 2.0))).astype(np.float32)
        return np.concatenate([local[..., :2], probe[..., None], pi[..., None]], axis=2)
