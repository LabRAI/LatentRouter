from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from latentrouter.cost import CostEstimator
from latentrouter.data import RoutingData
from latentrouter.evidence import CalibrationEvidence
from latentrouter.losses import outcome_nll, ranking_losses, regression_loss
from latentrouter.network import LatentRouterNet
from latentrouter.profiles import calibration_profiles, calibration_rows


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class LatentRouter:
    """LatentRouter: counterfactual utility prediction over a pool of candidate MLLMs.

    The capability token of model m_i is built from its calibration profile p_i (including its normalized
    latency l_i when latencies are given), its normalized average training cost c_i and the query-specific
    calibration statistics b_i(x), whose last entry is the calibrated estimate pi_i(x). After latent
    communication with the routing capsules, the outcome head predicts mu_i = sigmoid(logit pi_i(x) + g(a_i)),
    the bounded capsule correction gives mu_tilde_i, and the routing score is
    s_i(x) = mu_tilde_i(x) - lambda * c_hat_i(x).
    """

    def __init__(
        self,
        model_ids: list[str],
        seed: int = 0,
        *,
        hidden_dim: int = 96,
        num_query_tokens: int = 6,
        num_capsules: int = 7,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_bound: float = 0.05,
        feedback_temperature: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 16,
        patience: int = 4,
        pairwise_weight: float = 0.35,
        listwise_weight: float = 0.65,
        utility_weight: float = 0.25,
        residual_weight: float = 0.05,
        listwise_temperature: float = 0.05,
        cost_weight: float = 0.0,
        calibration_fraction: float = 0.35,
        max_calibration_samples: int | None = 4096,
        num_neighbors: int = 32,
        shrinkage: float = 5.0,
        probe_alpha: float = 1e4,
        probe_folds: int = 5,
        cost_alpha: float = 1e4,
        device: str = "auto",
    ) -> None:
        self.model_ids = list(model_ids)
        self.seed = int(seed)
        self.hidden_dim = int(hidden_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.num_capsules = int(num_capsules)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.residual_bound = float(residual_bound)
        self.feedback_temperature = float(feedback_temperature)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.pairwise_weight = float(pairwise_weight)
        self.listwise_weight = float(listwise_weight)
        self.utility_weight = float(utility_weight)
        self.residual_weight = float(residual_weight)
        self.listwise_temperature = float(listwise_temperature)
        self.cost_weight = float(cost_weight)
        self.calibration_fraction = float(calibration_fraction)
        self.max_calibration_samples = None if max_calibration_samples is None else int(max_calibration_samples)
        self.num_neighbors = int(num_neighbors)
        self.shrinkage = float(shrinkage)
        self.probe_alpha = float(probe_alpha)
        self.probe_folds = int(probe_folds)
        self.cost_alpha = float(cost_alpha)
        self.device = str(device)

    def fit(self, train: RoutingData, val: RoutingData | None = None) -> "LatentRouter":
        self.device_ = _device(self.device)
        torch.manual_seed(self.seed)
        if self.device_ == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        quality = train.quality.astype(np.float32)
        cost = train.cost.astype(np.float32)
        available = train.availability

        self.query_scaler_ = StandardScaler().fit(train.features)
        rows = calibration_rows(len(train), fraction=self.calibration_fraction, max_samples=self.max_calibration_samples, seed=self.seed)
        profiles = calibration_profiles(self.model_ids, quality, cost, available, train.slices, rows, train.latency)
        self.profile_scaler_ = StandardScaler().fit(profiles)
        self.profiles_ = self.profile_scaler_.transform(profiles).astype(np.float32)
        self.cost_scale_ = float(np.nanmax(cost[available])) if available.any() else 1.0
        self.mean_cost_ = (np.nanmean(np.where(available, cost, np.nan), axis=0) / max(self.cost_scale_, 1e-12)).astype(np.float32)
        self.evidence_ = CalibrationEvidence(self.num_neighbors, self.shrinkage, self.probe_alpha, self.probe_folds, self.seed)
        self.evidence_.fit(self.query_scaler_.transform(train.features), quality, available)

        self.net_ = LatentRouterNet(
            query_dim=train.features.shape[1],
            model_dim=self.profiles_.shape[1] + 5,
            hidden_dim=self.hidden_dim,
            num_query_tokens=self.num_query_tokens,
            num_capsules=self.num_capsules,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            residual_bound=self.residual_bound,
            feedback_temperature=self.feedback_temperature,
        ).to(self.device_)
        self._train(train, val)
        self.cost_estimator_ = CostEstimator(self.cost_alpha).fit(train)
        return self

    def _inputs(self, data: RoutingData, training: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_scaler_.transform(data.features)
        n = len(data)
        model_features = np.concatenate(
            [
                np.repeat(self.profiles_[None], n, axis=0),
                np.repeat(self.mean_cost_[None, :, None], n, axis=0),
                self.evidence_.transform(query, training=training),
            ],
            axis=2,
        ).astype(np.float32)
        model_features[~data.availability] = 0.0
        device = torch.device(self.device_)
        return (
            torch.from_numpy(query.astype(np.float32)).to(device),
            torch.from_numpy(model_features).to(device),
            torch.from_numpy(data.availability).to(device),
        )

    def _utility(self, data: RoutingData) -> tuple[np.ndarray, np.ndarray]:
        available = data.availability
        cost = np.zeros(data.cost.shape, dtype=np.float32)
        cost[available] = (data.cost.astype(np.float32)[available] / max(self.cost_scale_, 1e-12)).astype(np.float32)
        utility = np.full(data.quality.shape, -1e6, dtype=np.float32)
        utility[available] = (data.quality.astype(np.float32)[available] - self.cost_weight * cost[available]).astype(np.float32)
        return utility, cost

    def _loss(self, out: dict, quality: torch.Tensor, utility: torch.Tensor, cost: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        score = (out["mu_tilde"] - self.cost_weight * cost).masked_fill(~available, -1e6)
        pairwise, listwise = ranking_losses(score, utility, available, self.listwise_temperature)
        residual = out["delta"][available].pow(2).mean() if available.any() else score.new_tensor(0.0)
        return (
            outcome_nll(out["logit"], quality, available)
            + self.pairwise_weight * pairwise
            + self.listwise_weight * listwise
            + self.utility_weight * regression_loss(score, utility, available)
            + self.residual_weight * residual
        )

    def _train(self, train: RoutingData, val: RoutingData | None) -> None:
        device = torch.device(self.device_)
        query, model_features, available = self._inputs(train, training=True)
        utility, cost = self._utility(train)
        quality = torch.from_numpy(np.where(train.availability, train.quality.astype(np.float32), 0.0).astype(np.float32)).to(device)
        utility, cost = torch.from_numpy(utility).to(device), torch.from_numpy(cost).to(device)
        val_inputs = self._inputs(val) if val is not None else None

        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_state, best_score, stale = None, -1e18, 0
        self.epochs_trained_ = 0
        for _ in range(self.epochs):
            order = rng.permutation(len(train))
            self.net_.train()
            for start in range(0, len(train), self.batch_size):
                idx = torch.as_tensor(order[start : start + self.batch_size], dtype=torch.long, device=device)
                out = self.net_(query[idx], model_features[idx], available[idx])
                loss = self._loss(out, quality[idx], utility[idx], cost[idx], available[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                optimizer.step()
            self.epochs_trained_ += 1
            if val is None:
                continue
            score = self._selected_utility(val, val_inputs)
            if score > best_score + 1e-9:
                best_score, stale = score, 0
                best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()})
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_val_utility_ = best_score if best_state is not None else None

    def _forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_size: int | None = None) -> dict[str, np.ndarray]:
        query, model_features, available = inputs
        step = batch_size or max(len(query), 1)
        self.net_.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(query), step):
                out = self.net_(query[start : start + step], model_features[start : start + step], available[start : start + step])
                chunks.append({k: out[k].cpu().numpy().astype(np.float32) for k in ("mu", "sigma", "mu_tilde")})
        return {k: np.concatenate([c[k] for c in chunks]) for k in ("mu", "sigma", "mu_tilde")}

    def _selected_utility(self, data: RoutingData, inputs) -> float:
        scores = self._forward(inputs)["mu_tilde"]
        scores[~data.availability] = -1e6
        selected = np.where(data.availability, scores, -1e6).argmax(axis=1)
        utility, _ = self._utility(data)
        return float(np.nanmean(utility[np.arange(len(selected)), selected]))

    def predict(self, data: RoutingData, batch_size: int | None = None) -> dict[str, np.ndarray]:
        """Predicted outcome mean mu, standard deviation sigma and corrected mean mu_tilde, each (n, K)."""
        out = self._forward(self._inputs(data), batch_size)
        out["mu_tilde"][~data.availability] = -1e6
        return out

    def estimate_cost(self, data: RoutingData) -> np.ndarray:
        return self.cost_estimator_.predict(data)

    def route(self, data: RoutingData, lam: float = 0.0, mu_tilde: np.ndarray | None = None, cost: np.ndarray | None = None) -> np.ndarray:
        """Index of the selected model: argmax over available models of mu_tilde_i(x) - lam * c_hat_i(x) / c_max."""
        mu_tilde = self.predict(data)["mu_tilde"] if mu_tilde is None else mu_tilde
        cost = self.estimate_cost(data) if cost is None else cost
        scores = mu_tilde - lam * cost / self.cost_estimator_.scale_
        return np.where(data.availability, scores, -np.inf).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        device = self.device_
        self.net_.cpu()
        self.device_ = "cpu"
        with path.open("wb") as f:
            pickle.dump(self, f)
        self.device_ = device
        self.net_.to(device)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "LatentRouter":
        with Path(path).open("rb") as f:
            router = pickle.load(f)
        router.device_ = _device(device)
        router.net_.to(router.device_)
        return router
