from __future__ import annotations

import numpy as np
import pandas as pd


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def best_single_model(single_model_frontier: pd.DataFrame) -> pd.Series:
    if single_model_frontier.empty:
        return pd.Series({"model_id": None, "quality": np.nan, "cost": np.nan})
    return single_model_frontier.sort_values(["quality", "cost"], ascending=[False, True]).iloc[0]


def frontier_envelope(frontier: pd.DataFrame) -> pd.DataFrame:
    ordered = (
        frontier.sort_values(["mean_cost", "mean_quality", "lambda_value"], ascending=[True, False, True])
        .groupby("mean_cost", as_index=False)
        .agg(
            mean_quality=("mean_quality", "max"),
            lambda_value=("lambda_value", "first"),
        )
        .sort_values("mean_cost")
        .reset_index(drop=True)
    )
    ordered["frontier_quality"] = np.maximum.accumulate(ordered["mean_quality"].to_numpy())
    return ordered


def compute_nauc(frontier: pd.DataFrame) -> float:
    envelope = frontier_envelope(frontier)
    if envelope.empty:
        return 0.0
    if len(envelope) == 1:
        return float(envelope["frontier_quality"].iloc[0])
    costs = envelope["mean_cost"].to_numpy(dtype=float)
    qualities = envelope["frontier_quality"].to_numpy(dtype=float)
    cost_range = costs.max() - costs.min()
    if cost_range <= 0:
        return float(qualities.max())
    normalized_costs = (costs - costs.min()) / cost_range
    return _trapezoid(qualities, normalized_costs)


def compute_ps(frontier: pd.DataFrame) -> float:
    envelope = frontier_envelope(frontier)
    if envelope.empty:
        return 0.0
    return float(envelope["frontier_quality"].max())


def compute_qnc(
    frontier: pd.DataFrame,
    single_model_frontier: pd.DataFrame,
    quality_fraction: float = 1.0,
    quality_epsilon: float = 0.0,
) -> float:
    envelope = frontier_envelope(frontier)
    if envelope.empty or single_model_frontier.empty:
        return float("nan")

    best_single = best_single_model(single_model_frontier)
    threshold = float(best_single["quality"]) * float(quality_fraction) - float(quality_epsilon)
    threshold = max(threshold, 0.0)
    baseline_cost = float(best_single["cost"])
    candidates = envelope.loc[envelope["frontier_quality"] >= threshold, "mean_cost"]
    if candidates.empty or baseline_cost <= 0:
        return float("inf")
    return float(candidates.min() / baseline_cost)
