from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from latentrouter.benchmarks.registry import load_benchmark_protocol
from latentrouter.config import EvaluationConfig
from latentrouter.embedding.store import load_router_bundle
from latentrouter.schemas import RouterDatasetBundle


def evaluate_router(
    router_name: str,
    predicted_utilities: np.ndarray,
    bundle: RouterDatasetBundle,
    config: EvaluationConfig,
    run_dir: str | Path,
    prediction_wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    protocol = load_benchmark_protocol(protocol_id=bundle.protocol_id, benchmark_id=bundle.benchmark_id)
    return protocol.evaluate(
        router_name=router_name,
        predicted_utilities=predicted_utilities,
        bundle=bundle,
        config=config,
        run_dir=run_dir,
        prediction_wall_time_seconds=prediction_wall_time_seconds,
    )


def run_router_on_benchmark(
    router_name: str,
    router,
    processed_dir: str | Path,
    config: EvaluationConfig,
    run_dir: str | Path,
    split: str = "test",
    feature_dir: str | Path | None = None,
) -> dict[str, Any]:
    bundle = load_router_bundle(processed_dir, split=split, feature_dir=feature_dir)
    started = time.perf_counter()
    predicted_utilities = router.predict_utilities(bundle)
    elapsed = time.perf_counter() - started
    return evaluate_router(
        router_name=router_name,
        predicted_utilities=predicted_utilities,
        bundle=bundle,
        config=config,
        run_dir=run_dir,
        prediction_wall_time_seconds=elapsed if config.measure_throughput else None,
    )
