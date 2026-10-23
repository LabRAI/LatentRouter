from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from latentrouter import LatentRouter, RoutingData
from latentrouter.benchmarks.mmr_bench import curve_metrics, threshold_curve
from latentrouter.benchmarks.vl_routerbench import rank_score
from latentrouter.evidence import CalibrationEvidence


def synthetic(n: int = 240, k: int = 5, d: int = 12, seed: int = 0) -> RoutingData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    skill = rng.normal(size=(d, k))
    quality = (x @ skill + rng.normal(size=(n, k)) > 0).astype(float)
    price = np.linspace(0.1, 1.0, k)
    tokens = rng.integers(10, 200, size=(n, k)).astype(float)
    input_cost = np.full((n, k), 0.01)
    return RoutingData(
        sample_ids=np.arange(n),
        model_ids=[f"model-{i}b" for i in range(k)],
        features=x,
        quality=quality,
        cost=input_cost + tokens * price * 1e-3,
        slices=pd.DataFrame({"dataset": rng.choice(["ocr", "chart", "math"], size=n)}),
        input_cost=input_cost,
        output_tokens=tokens,
        output_price=price * 1e-3,
    )


def small_router(data: RoutingData, **kwargs) -> LatentRouter:
    params = dict(hidden_dim=16, num_query_tokens=2, num_capsules=3, num_heads=2, batch_size=64, epochs=2, num_neighbors=8, device="cpu")
    params.update(kwargs)
    return LatentRouter(data.model_ids, seed=0, **params)


def test_fit_predict_route():
    data = synthetic()
    train, val = data.subset(np.arange(200)), data.subset(np.arange(200, 240))
    router = small_router(train).fit(train, val)
    out = router.predict(val)
    assert out["mu"].shape == out["mu_tilde"].shape == (40, 5)
    assert np.all((out["mu_tilde"] >= 0) & (out["mu_tilde"] <= 1))
    assert router.estimate_cost(val).shape == (40, 5)
    val.availability[:, 0] = False
    assert not np.any(router.route(val, lam=0.0) == 0)


def test_outcome_head_starts_from_calibration_estimate():
    data = synthetic()
    router = small_router(data, epochs=0, residual_bound=0.0).fit(data)
    pi = router.evidence_.transform(router.query_scaler_.transform(data.features))[..., -1]
    np.testing.assert_allclose(router.predict(data)["mu"], np.clip(pi, 1e-3, 1 - 1e-3), atol=1e-6)


def test_training_queries_exclude_their_own_outcome():
    data = synthetic()
    flipped = data.quality.copy()
    flipped[0] = 1.0 - flipped[0]
    a = CalibrationEvidence(num_neighbors=8).fit(data.features, data.quality, data.availability)
    b = CalibrationEvidence(num_neighbors=8).fit(data.features, flipped, data.availability)
    local_and_probe = [0, 2]
    np.testing.assert_array_equal(a.transform(data.features, training=True)[0][:, local_and_probe], b.transform(data.features, training=True)[0][:, local_and_probe])


def test_save_load(tmp_path):
    data = synthetic()
    router = small_router(data).fit(data)
    router.save(tmp_path / "router.pkl")
    loaded = LatentRouter.load(tmp_path / "router.pkl", device="cpu")
    np.testing.assert_array_equal(router.predict(data)["mu_tilde"], loaded.predict(data)["mu_tilde"])


def _reference_curve(perf, cost, quality, realized):
    points = []
    for c in np.unique(cost):
        chosen = []
        for r in range(len(perf)):
            ok = np.where(cost[r] <= c)[0]
            chosen.append(ok[np.argmax(perf[r, ok])] if len(ok) else 0)
        rows = np.arange(len(perf))
        points.append({"cost": realized[rows, chosen].mean(), "performance": quality[rows, chosen].mean()})
    return points


def test_threshold_curve_matches_reference():
    rng = np.random.default_rng(1)
    perf, cost = rng.random((50, 4)), rng.random((50, 4))
    quality, realized = (rng.random((50, 4)) > 0.5).astype(float), rng.random((50, 4))
    fast = curve_metrics(threshold_curve(perf, cost, quality, realized), quality, realized)
    slow = curve_metrics(_reference_curve(perf, cost, quality, realized), quality, realized)
    assert fast == pytest.approx(slow)


def test_rank_score_bounds():
    assert rank_score(1.0, 1e-5, 1e-5, 1e-3) == pytest.approx(1.0, abs=1e-6)
    assert rank_score(0.8, 1e-3, 1e-5, 1e-3) == 0.0
