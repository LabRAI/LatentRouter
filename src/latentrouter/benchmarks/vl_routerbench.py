from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from latentrouter.data import RoutingData
from latentrouter.router import LatentRouter

TEXT_FEATURES = "text/bge-m3.parquet"
IMAGE_FEATURES = "vision/siglip-large-patch16-384.parquet"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


def _cost_matrix(root: Path, meta: pd.DataFrame) -> np.ndarray:
    matrices = root / "data" / "matrices"
    if not (matrices / "C.npy").exists():
        return np.load(matrices / "C.npz")["C"]
    full = np.load(matrices / "C.npy")
    if not (matrices / "sample_ids.pkl").exists():
        return full
    with open(matrices / "sample_ids.pkl", "rb") as f:
        position = {sid: i for i, sid in enumerate(pickle.load(f))}
    return np.stack([full[position[sid]] if sid in position else full.mean(axis=0) for sid in meta["sample_id"]])


def load(root: str | Path, features: str | Path | None = None) -> dict:
    """Train / dev / test RoutingData from the artifacts of the official VL-RouterBench pipeline.

    Query features are the L2-normalized BGE-M3 question embedding concatenated with the L2-normalized
    SigLIP image embedding, standardized with statistics of the training split.
    """
    root = Path(root)
    features = Path(features) if features is not None else root / "EMBEDDINGS"
    meta = pd.read_parquet(root / "data" / "registry" / "meta.parquet")
    quality = np.load(root / "data" / "matrices" / "Y.npz")["Y"]
    cost = _cost_matrix(root, meta)
    with open(root / "data" / "registry" / "model_index.pkl", "rb") as f:
        models = list(pickle.load(f))
    emb = pd.read_parquet(features / TEXT_FEATURES).merge(pd.read_parquet(features / IMAGE_FEATURES), on="sample_id", suffixes=("_text", "_image"))
    emb = emb.merge(meta[["sample_id"]], on="sample_id").set_index("sample_id").reindex(meta["sample_id"]).reset_index()
    text, image = np.vstack(emb["embedding_text"].to_numpy()), np.vstack(emb["embedding_image"].to_numpy())
    meta_pos = {sid: i for i, sid in enumerate(meta["sample_id"])}
    emb_pos = {sid: i for i, sid in enumerate(emb["sample_id"])}

    details = pd.read_csv(root / "data" / "matrices" / "cost_details.csv")
    generated = details[details["output_tokens"] > 0]
    price = (generated["output_cost_usd"] / generated["output_tokens"]).groupby(generated["model"]).median().reindex(models).to_numpy(float)
    output_tokens = details.pivot(index="sample_id", columns="model", values="output_tokens")[models]
    input_cost = details.pivot(index="sample_id", columns="model", values="input_cost_usd")[models]
    bounds = json.loads((root / "data" / "matrices" / "cost_bounds.json").read_text())

    parts = {}
    for split in ("train", "dev", "test"):
        ids = sorted(set(meta.loc[meta["split"] == split, "sample_id"]) & set(emb["sample_id"]))
        m = np.array([meta_pos[s] for s in ids])
        e = np.array([emb_pos[s] for s in ids])
        parts[split] = RoutingData(
            sample_ids=np.array(ids),
            model_ids=models,
            features=np.hstack([_l2_normalize(text[e]), _l2_normalize(image[e])]),
            quality=quality[m].astype(float),
            cost=cost[m].astype(float),
            slices=pd.DataFrame({"group": meta["dataset"].to_numpy()[m], "dataset": meta["dataset"].to_numpy()[m]}),
            input_cost=input_cost.reindex(ids).to_numpy(float),
            output_tokens=output_tokens.reindex(ids).to_numpy(float),
            output_price=price,
        )
    scaler = StandardScaler().fit(parts["train"].features)
    for part in parts.values():
        part.features = scaler.transform(part.features).astype(np.float32)
    return {**parts, "scaler": scaler, "cmin": float(bounds["cmin"]), "cmax": float(bounds["cmax"])}


def rank_score(accuracy: float, cost: float, cmin: float, cmax: float, beta: float = 0.1, eps: float = 1e-12) -> float:
    c = (np.log2(cmax) - np.log2(float(np.clip(cost, cmin, cmax)))) / (np.log2(cmax) - np.log2(cmin) + eps)
    c = float(np.clip(c, 0.0, 1.0))
    a = float(np.clip(accuracy, 0.0, 1.0))
    return float(np.clip((1 + beta) * a * c / (beta * a + c + eps), 0.0, 1.0))


def metrics(choice: np.ndarray, data: RoutingData, cmin: float, cmax: float) -> dict:
    rows = np.arange(len(choice))
    accuracy = float(data.quality[rows, choice].mean())
    cost = float(data.cost[rows, choice].mean())
    return {"accuracy": accuracy, "avg_cost": cost, "rank_score": rank_score(accuracy, cost, cmin, cmax)}


def run(config: dict, seed: int) -> tuple[LatentRouter, dict]:
    """Train on the official train split with early stopping on dev; lambda is selected on dev, by Rank Score
    for the performance-cost setting and by accuracy for the performance-oriented setting."""
    data = load(config["data"]["root"], config["data"].get("features"))
    router = LatentRouter(data["train"].model_ids, seed, **config.get("router", {})).fit(data["train"], data["dev"])
    scored = {}
    for split in ("dev", "test"):
        scored[split] = (np.asarray(router.predict(data[split])["mu_tilde"], dtype=float), router.estimate_cost(data[split]))
    trials = []
    for lam in config.get("lambdas", [0.0]):
        trial = {"lambda": lam}
        for split, (mu_tilde, cost) in scored.items():
            choice = router.route(data[split], lam, mu_tilde=mu_tilde, cost=cost)
            trial[split] = metrics(choice, data[split], data["cmin"], data["cmax"])
        trials.append(trial)
    by_rs = max(trials, key=lambda t: t["dev"]["rank_score"])
    by_acc = max(trials, key=lambda t: t["dev"]["accuracy"])
    return router, {
        "benchmark": "vl_routerbench",
        "seed": seed,
        "performance_cost": {"lambda": by_rs["lambda"], "test": by_rs["test"]},
        "performance_oriented": {"lambda": by_acc["lambda"], "test": by_acc["test"]},
        "trials": trials,
    }
