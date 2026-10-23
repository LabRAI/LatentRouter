from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from latentrouter.data import RoutingData
from latentrouter.router import LatentRouter

IMAGE_DIRS = {
    "MathVerse": "MathVerse",
    "MathVision": "MathVision",
    "MathVista": "MathVista",
    "OCRBench": "OCRBench",
    "RealWorldQA": "RealWorldQA",
    "MMStar": "MMStar",
    "SEEDBench2_Plus": "SEEDBenchv2Plus",
}


def _prefix(dataset_idx: str) -> str | None:
    s = str(dataset_idx)
    matches = [p for p in IMAGE_DIRS if s == p or s.startswith(p + "_")]
    return max(matches, key=len) if matches else None


def image_path(root: Path, dataset_idx: str) -> str:
    prefix = _prefix(dataset_idx)
    if prefix is None:
        return ""
    rest = str(dataset_idx)
    rest = rest[len(prefix) + 1 :] if rest.startswith(prefix + "_") else rest.split("_")[-1]
    base = (root / IMAGE_DIRS[prefix]).resolve()
    for ext in (".jpg", ".png", ".jpeg", ".webp"):
        if (base / f"{rest}{ext}").exists():
            return str(base / f"{rest}{ext}")
    return str(base / f"{rest}.jpg")


def _correctness(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if series.dtype.kind in "fiu":
        return pd.to_numeric(series, errors="coerce").astype(float)
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "yes": 1.0, "no": 0.0}
    return series.astype(str).str.strip().str.lower().map(mapping).astype(float)


def table_path(root: str | Path) -> Path:
    root = Path(root)
    return root / "MMR_Bench.csv" if (root / "MMR_Bench.csv").exists() else root / "MMR-Bench.csv"


def load_table(root: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """MMR-Bench outcome table with missing outcomes filled by column means, as in the official benchmark."""
    raw = pd.read_csv(table_path(root))
    prefixes = [_prefix(i) for i in raw["dataset_idx"]]
    keep = np.array([p is not None for p in prefixes])
    raw = raw[keep].copy()
    models = sorted({c[: -len("_correct")] for c in raw.columns if c.endswith("_correct")}
                    & {c[: -len("_cost")] for c in raw.columns if c.endswith("_cost")})
    table = pd.DataFrame(index=raw.index)
    table["dataset"] = [IMAGE_DIRS[p] for p in prefixes if p is not None]
    table["dataset_idx"] = raw["dataset_idx"].astype(str)
    table["question"] = raw["question"].astype(str)
    for m in models:
        table[f"{m}_quality"] = _correctness(raw[f"{m}_correct"])
        table[f"{m}_cost"] = pd.to_numeric(raw[f"{m}_cost"], errors="coerce")
    for col in [f"{m}_{k}" for m in models for k in ("quality", "cost")]:
        if table[col].isna().any() and table[col].notna().any():
            table[col] = table[col].fillna(float(table[col].dropna().mean()))
    return table, models


def load_features(path: str | Path, table: pd.DataFrame) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    index = {d: i for i, d in enumerate(z["dataset_idx"].astype(str))}
    rows = np.array([index[d] for d in table["dataset_idx"]])
    return np.concatenate([z["text"][rows], z["image"][rows]], axis=1).astype(np.float32)


def routing_data(part: pd.DataFrame, table: pd.DataFrame, features: np.ndarray, models: list[str]) -> RoutingData:
    return RoutingData(
        sample_ids=part.index.astype(str).to_numpy(),
        model_ids=models,
        features=features[table.index.get_indexer(part.index)],
        quality=part[[f"{m}_quality" for m in models]].to_numpy(dtype=float),
        cost=part[[f"{m}_cost" for m in models]].to_numpy(dtype=float),
        slices=pd.DataFrame({"group": part["dataset"].to_numpy(), "dataset": part["dataset"].to_numpy()}),
    )


def threshold_curve(perf: np.ndarray, cost: np.ndarray, quality: np.ndarray, realized_cost: np.ndarray) -> list[dict]:
    """Official MMR-Bench operating points: for every cost threshold C taken from the predicted costs, each query
    goes to the model with the highest predicted performance among models with predicted cost <= C (model 0 if
    none qualifies). Each point is the mean realized cost and quality of the chosen models."""
    n, k = perf.shape
    rows = np.arange(n)
    order = np.argsort(cost, axis=1, kind="stable")
    best = np.zeros((n, k), dtype=np.int64)
    cur, cur_perf = np.full(n, -1, dtype=np.int64), np.full(n, -np.inf)
    for j in range(k):
        cand = order[:, j]
        p = perf[rows, cand]
        better = (p > cur_perf) | ((p == cur_perf) & (cand < cur))
        cur, cur_perf = np.where(better, cand, cur), np.where(better, p, cur_perf)
        best[:, j] = cur
    thresholds = np.take_along_axis(cost, order, axis=1).ravel()
    event_row, event_model, event_rank = np.repeat(rows, k), best.ravel(), np.tile(np.arange(k), n)
    o = np.lexsort((event_rank, thresholds))
    thresholds, event_row, event_model = thresholds[o], event_row[o], event_model[o]
    chosen = np.zeros(n, dtype=np.int64)
    total_q, total_c = float(quality[rows, 0].sum()), float(realized_cost[rows, 0].sum())
    points, e = [], 0
    while e < len(thresholds):
        c = thresholds[e]
        while e < len(thresholds) and thresholds[e] == c:
            r, new = event_row[e], event_model[e]
            if new != chosen[r]:
                total_q += quality[r, new] - quality[r, chosen[r]]
                total_c += realized_cost[r, new] - realized_cost[r, chosen[r]]
                chosen[r] = new
            e += 1
        points.append({"cost": total_c / n, "performance": total_q / n})
    return points


def curve_metrics(points: list[dict], quality: np.ndarray, realized_cost: np.ndarray) -> dict:
    """nAUC, Ps and QNC of a set of operating points, following the official MMR-Bench evaluator."""
    pareto, best = [], -math.inf
    for p in sorted(points, key=lambda p: (p["cost"], -p["performance"])):
        if p["performance"] > best:
            pareto.append(p)
            best = p["performance"]
    xs = np.array([p["cost"] for p in pareto], dtype=float)
    ys = np.array([p["performance"] for p in pareto], dtype=float)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapezoid(ys[np.argsort(xs)], np.sort(xs))) if len(xs) >= 2 else float("nan")
    lo, hi = (float(xs.min()), float(xs.max())) if len(xs) else (float("nan"), float("nan"))
    nauc = auc / (hi - lo) if not math.isnan(auc) and hi > lo else float("nan")
    accuracy, mean_cost = np.nanmean(quality, axis=0), np.nanmean(realized_cost, axis=0)
    top = int(np.argmax(accuracy))
    qnc = next((p["cost"] / mean_cost[top] if mean_cost[top] > 0 else float("inf") for p in pareto
                if p["performance"] >= accuracy[top]), float("inf"))
    return {"nAUC": nauc, "Ps": float(ys.max()) if len(ys) else float("nan"), "QNC": float(qnc)}


def evaluate(router, data: RoutingData) -> dict:
    perf = np.asarray(router.predict(data)["mu_tilde"], dtype=float)
    points = threshold_curve(perf, router.estimate_cost(data), data.quality, data.cost)
    return curve_metrics(points, data.quality, data.cost)


def run(config: dict, seed: int) -> tuple[LatentRouter, dict]:
    """Official split (train_fraction of the queries for training, the rest for test); hyperparameters in
    config["search"] are selected by nAUC on a validation part of the training split."""
    data_cfg = config["data"]
    table, models = load_table(data_cfg["root"])
    features = load_features(data_cfg["features"], table)
    train, test = train_test_split(table, train_size=float(data_cfg.get("train_fraction", 0.2)), shuffle=True, random_state=seed)
    fit, val = train_test_split(train, train_size=1.0 - float(data_cfg.get("val_fraction", 0.25)), shuffle=True, random_state=seed)
    fit_data, val_data, test_data = (routing_data(part, table, features, models) for part in (fit, val, test))

    search = config.get("search") or {}
    trials, best = [], None
    for values in itertools.product(*search.values()):
        params = dict(zip(search, values))
        router = LatentRouter(models, seed, **{**config.get("router", {}), **params}).fit(fit_data, val_data)
        val_metrics = evaluate(router, val_data)
        trials.append({"params": params, "val": val_metrics})
        score = -np.inf if math.isnan(val_metrics["nAUC"]) else val_metrics["nAUC"]
        if best is None or score > best[0]:
            best = (score, params, router)
    _, selected, router = best
    return router, {"benchmark": "mmr_bench", "seed": seed, "selected": selected, "test": evaluate(router, test_data), "trials": trials}
