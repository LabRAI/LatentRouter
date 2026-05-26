from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from latentrouter.config import EvaluationConfig
from latentrouter.evaluation.runner import evaluate_router
from latentrouter.io import ensure_dir, write_json
from latentrouter.schemas import RouterDatasetBundle


@dataclass(frozen=True, slots=True)
class PoolShiftSpec:
    name: str
    model_ids: list[str]
    description: str


def subset_bundle_models(bundle: RouterDatasetBundle, model_ids: list[str]) -> RouterDatasetBundle:
    lookup = {model_id: idx for idx, model_id in enumerate(bundle.model_ids)}
    missing = [model_id for model_id in model_ids if model_id not in lookup]
    if missing:
        raise ValueError(f"Requested model IDs are not present in bundle: {missing!r}")
    indices = np.asarray([lookup[model_id] for model_id in model_ids], dtype=int)
    return RouterDatasetBundle(
        sample_ids=bundle.sample_ids.copy(),
        model_ids=[bundle.model_ids[idx] for idx in indices],
        features=bundle.features.copy(),
        correctness=bundle.correctness[:, indices].copy(),
        costs=bundle.costs[:, indices].copy(),
        token_counts=None if bundle.token_counts is None else bundle.token_counts[:, indices].copy(),
        availability=bundle.availability[:, indices].copy(),
        sample_frame=bundle.sample_frame.copy(),
        benchmark_id=bundle.benchmark_id,
        protocol_id=bundle.protocol_id,
        evaluation_metadata={**dict(bundle.evaluation_metadata or {}), "pool_shift_model_ids": [bundle.model_ids[idx] for idx in indices]},
    )


def _mean_quality(bundle: RouterDatasetBundle) -> np.ndarray:
    quality = np.where(bundle.availability, bundle.correctness, np.nan)
    return np.nanmean(quality, axis=0)


def _mean_cost(bundle: RouterDatasetBundle) -> np.ndarray:
    costs = np.where(bundle.availability, bundle.costs, np.nan)
    return np.nanmean(costs, axis=0)


def build_removed_model_specs(
    bundle: RouterDatasetBundle,
    *,
    seed: int = 20260308,
    random_remove_fraction: float = 0.3,
) -> list[PoolShiftSpec]:
    model_ids = list(bundle.model_ids)
    if len(model_ids) <= 1:
        return [PoolShiftSpec("full_pool", model_ids, "Full model pool.")]
    quality = _mean_quality(bundle)
    costs = _mean_cost(bundle)
    strongest = model_ids[int(np.nanargmax(quality))]
    cheapest = model_ids[int(np.nanargmin(costs))]
    rng = np.random.default_rng(seed)
    specs = [
        PoolShiftSpec("full_pool", model_ids, "Full model pool."),
        PoolShiftSpec(
            "remove_strongest",
            [model_id for model_id in model_ids if model_id != strongest],
            f"Removed strongest calibration/test model: {strongest}.",
        ),
        PoolShiftSpec(
            "remove_cheapest",
            [model_id for model_id in model_ids if model_id != cheapest],
            f"Removed cheapest calibration/test model: {cheapest}.",
        ),
    ]
    remove_one = str(rng.choice(model_ids))
    specs.append(
        PoolShiftSpec(
            "remove_random_one",
            [model_id for model_id in model_ids if model_id != remove_one],
            f"Removed deterministic random model: {remove_one}.",
        )
    )
    for fraction, name in [(0.3, "remove_30pct"), (0.5, "remove_50pct"), (random_remove_fraction, "remove_random_fraction")]:
        count = min(max(1, int(round(len(model_ids) * fraction))), len(model_ids) - 1)
        removed = set(rng.choice(model_ids, size=count, replace=False).tolist())
        specs.append(
            PoolShiftSpec(
                name,
                [model_id for model_id in model_ids if model_id not in removed],
                f"Removed {count} deterministic random models: {sorted(removed)}.",
            )
        )
    for model_id in model_ids:
        specs.append(
            PoolShiftSpec(
                f"leave_one_out__{model_id}",
                [candidate for candidate in model_ids if candidate != model_id],
                f"Leave-one-model-out removal for {model_id}.",
            )
        )
    deduped: dict[tuple[str, tuple[str, ...]], PoolShiftSpec] = {}
    for spec in specs:
        deduped[(spec.name, tuple(spec.model_ids))] = spec
    return list(deduped.values())


def evaluate_pool_shift_specs(
    *,
    router_name: str,
    router: Any,
    test_bundle: RouterDatasetBundle,
    specs: list[PoolShiftSpec],
    run_dir: str | Path,
    config: EvaluationConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or EvaluationConfig(num_lambdas=201, aggregate_by=["dataset_name", "mode_id"])
    run_dir = ensure_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        spec_dir = ensure_dir(run_dir / spec.name)
        shifted = subset_bundle_models(test_bundle, spec.model_ids)
        try:
            started = time.perf_counter()
            utilities = router.predict_utilities(shifted)
            elapsed = time.perf_counter() - started
            evaluation = evaluate_router(
                router_name=router_name,
                predicted_utilities=utilities,
                bundle=shifted,
                config=config,
                run_dir=spec_dir,
                prediction_wall_time_seconds=elapsed if config.measure_throughput else None,
            )
            metrics = dict(evaluation["metrics"])
            status = "completed"
            error = ""
        except Exception as exc:  # pragma: no cover - exercised by script-level skip paths
            metrics = {}
            status = "failed"
            error = str(exc)
            write_json(spec_dir / "error.json", {"status": status, "error": error, "pool_shift": spec.name})
        row = {
            "router_name": router_name,
            "pool_shift": spec.name,
            "status": status,
            "description": spec.description,
            "num_models": len(spec.model_ids),
            "model_ids": spec.model_ids,
            "error": error,
            "prediction_ms_per_sample": float(1000.0 * elapsed / max(len(shifted.sample_ids), 1)) if status == "completed" else float("nan"),
            **metrics,
        }
        rows.append(row)
    write_json(run_dir / "pool_shift_summary.json", {"router_name": router_name, "results": rows})
    return rows
