from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentrouter.config import EvaluationConfig
from latentrouter.embedding.store import load_router_bundle
from latentrouter.evaluation.runner import evaluate_router
from latentrouter.routers.paper_latent_communication_router import PaperSection32LatentCommunicationRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the original latent-communication router on VL-RouterBench.")
    parser.add_argument("--processed-dir", default="data/processed/vl_routerbench")
    parser.add_argument("--feature-dir", default="data/processed/vl_routerbench/openclip_features")
    parser.add_argument("--output-dir", default="artifacts/vl_routerbench_repro")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_router_bundle(args.processed_dir, split="train", feature_dir=args.feature_dir)
    valid = load_router_bundle(args.processed_dir, split="val", feature_dir=args.feature_dir)
    test = load_router_bundle(args.processed_dir, split="test", feature_dir=args.feature_dir)
    output_root = Path(args.output_dir)
    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in args.seeds:
        print(f"Training original router, seed={seed}", flush=True)
        router = PaperSection32LatentCommunicationRouter(
            model_ids=list(train.model_ids),
            random_seed=seed,
            setting="performance_oriented",
            hidden_dim=48,
            num_query_tokens=4,
            num_capsules=7,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            residual_bound=0.05,
            feedback_temperature=0.25,
            learning_rate=args.learning_rate,
            weight_decay=1e-4,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            calibration_fraction=0.35,
            max_calibration_samples=512,
            device=args.device,
        )
        router.fit(train, valid)
        model_path = model_root / f"oriented_{seed}.pkl"
        router.save(model_path)
        run_dir = output_root / f"seed_{seed}" / "baseline"
        evaluation = EvaluationConfig(
            lambda_min_exp=-4.0,
            lambda_max_exp=4.0,
            num_lambdas=51,
            output_dir=str(run_dir),
            aggregate_by=["dataset_name", "mode_id"],
            measure_throughput=True,
        )
        scores = router.predict_utilities(test)
        metrics = evaluate_router(
            router_name="original_from_scratch_latentrouter",
            predicted_utilities=scores,
            bundle=test,
            config=evaluation,
            run_dir=run_dir,
        )
        record = {
            "seed": seed,
            "model_path": str(model_path),
            "primary_metric": metrics["metrics"]["primary_metric"],
            "primary_metric_value": float(metrics["metrics"]["primary_metric_value"]),
            "avg_accuracy": float(metrics["metrics"]["avg_accuracy"]),
            "avg_cost": float(metrics["metrics"]["avg_cost"]),
            "fit_summary": router.fit_summary_,
        }
        rows.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    aggregate = {
        "seed_count": len(rows),
        "primary_metric": rows[0]["primary_metric"] if rows else None,
        "mean_primary_metric": sum(row["primary_metric_value"] for row in rows) / len(rows),
        "mean_avg_accuracy": sum(row["avg_accuracy"] for row in rows) / len(rows),
        "mean_avg_cost": sum(row["avg_cost"] for row in rows) / len(rows),
    }
    with (output_root / "baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": rows, "aggregate": aggregate}, handle, indent=2, sort_keys=True)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
