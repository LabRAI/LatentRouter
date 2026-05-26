from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latentrouter.config import EvaluationConfig
from latentrouter.io import ensure_dir, write_json
from latentrouter.schemas import RouterDatasetBundle


def lambda_grid(config: EvaluationConfig) -> np.ndarray:
    if config.num_lambdas < 2:
        return np.array([0.0], dtype=float)
    positive = np.logspace(config.lambda_min_exp, config.lambda_max_exp, config.num_lambdas - 1)
    return np.concatenate([[0.0], positive]).astype(float)


def single_model_frontier(bundle: RouterDatasetBundle) -> pd.DataFrame:
    rows = []
    for model_idx, model_id in enumerate(bundle.model_ids):
        valid = bundle.availability[:, model_idx]
        if not np.any(valid):
            continue
        row: dict[str, Any] = {
            "model_id": model_id,
            "cost": float(np.nanmean(bundle.costs[valid, model_idx])),
            "quality": float(np.nanmean(bundle.correctness[valid, model_idx])),
        }
        if bundle.token_counts is not None:
            row["token_count"] = float(np.nanmean(bundle.token_counts[valid, model_idx]))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["model_id", "cost", "quality"])
    return pd.DataFrame(rows).sort_values("cost").reset_index(drop=True)


def oracle_frontier(bundle: RouterDatasetBundle, lambdas: np.ndarray) -> pd.DataFrame:
    max_cost = float(np.nanmax(bundle.costs[bundle.availability])) if np.any(bundle.availability) else 1.0
    normalized_cost = np.where(bundle.availability, bundle.costs / max(max_cost, 1e-12), np.inf)
    oracle_rows = []
    for lambda_value in lambdas:
        penalized = np.full(bundle.correctness.shape, -np.inf, dtype=float)
        penalized[bundle.availability] = (
            bundle.correctness[bundle.availability]
            - lambda_value * normalized_cost[bundle.availability]
        )
        chosen = penalized.argmax(axis=1)
        mask = bundle.availability[np.arange(len(bundle.sample_ids)), chosen]
        row: dict[str, Any] = {
            "lambda_value": float(lambda_value),
            "mean_cost": float(np.nanmean(bundle.costs[np.arange(len(bundle.sample_ids))[mask], chosen[mask]])),
            "mean_quality": float(np.nanmean(bundle.correctness[np.arange(len(bundle.sample_ids))[mask], chosen[mask]])),
        }
        if bundle.token_counts is not None:
            row["mean_token_count"] = float(
                np.nanmean(bundle.token_counts[np.arange(len(bundle.sample_ids))[mask], chosen[mask]])
            )
        oracle_rows.append(row)
    return pd.DataFrame(oracle_rows)


def materialize_routes(
    predicted_utilities: np.ndarray,
    bundle: RouterDatasetBundle,
    lambdas: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_cost = float(np.nanmax(bundle.costs[bundle.availability])) if np.any(bundle.availability) else 1.0
    normalized_cost = np.where(bundle.availability, bundle.costs / max(max_cost, 1e-12), np.inf)

    frontier_rows: list[dict[str, Any]] = []
    route_frames: list[pd.DataFrame] = []
    row_indices = np.arange(len(bundle.sample_ids))
    model_names = np.array(bundle.model_ids, dtype=object)

    for lambda_value in lambdas:
        penalized = np.full(predicted_utilities.shape, -np.inf, dtype=float)
        penalized[bundle.availability] = (
            predicted_utilities[bundle.availability]
            - lambda_value * normalized_cost[bundle.availability]
        )
        chosen = penalized.argmax(axis=1)
        chosen_valid = bundle.availability[row_indices, chosen]
        chosen_rows = row_indices[chosen_valid]
        chosen_models = model_names[chosen]
        quality = bundle.correctness[row_indices, chosen]
        cost = bundle.costs[row_indices, chosen]

        route_frame = bundle.sample_frame.copy()
        route_frame["lambda_value"] = float(lambda_value)
        route_frame["selected_model"] = chosen_models
        route_frame["correctness"] = quality
        route_frame["cost"] = cost
        route_frame["normalized_cost"] = normalized_cost[row_indices, chosen]
        route_frame["predicted_utility"] = predicted_utilities[row_indices, chosen]
        if bundle.token_counts is not None:
            route_frame["token_count"] = bundle.token_counts[row_indices, chosen]
        route_frames.append(route_frame.loc[chosen_valid].copy())

        frontier_row: dict[str, Any] = {
            "lambda_value": float(lambda_value),
            "mean_cost": float(np.nanmean(cost[chosen_valid])),
            "mean_quality": float(np.nanmean(quality[chosen_valid])),
        }
        if bundle.token_counts is not None:
            frontier_row["mean_token_count"] = float(np.nanmean(bundle.token_counts[chosen_rows, chosen[chosen_valid]]))
        frontier_rows.append(frontier_row)

    routes = pd.concat(route_frames, axis=0, ignore_index=True)
    frontier = pd.DataFrame(frontier_rows).sort_values("mean_cost").reset_index(drop=True)
    return routes, frontier


def write_evaluation_outputs(
    run_dir: str | Path,
    metrics: dict[str, Any],
    frontier: pd.DataFrame,
    routes: pd.DataFrame,
    single_model: pd.DataFrame,
    oracle: pd.DataFrame,
    slice_metrics: pd.DataFrame,
) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    frontier.to_csv(run_dir / "frontier.csv", index=False)
    routes.to_parquet(run_dir / "routes.parquet", index=False)
    single_model.to_csv(run_dir / "single_model_frontier.csv", index=False)
    oracle.to_csv(run_dir / "oracle_frontier.csv", index=False)
    slice_metrics.to_csv(run_dir / "slice_metrics.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    return {
        "metrics": metrics,
        "frontier": frontier,
        "routes": routes,
        "single_model_frontier": single_model,
        "oracle_frontier": oracle,
        "slice_metrics": slice_metrics,
    }
