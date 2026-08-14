from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentrouter.experiments.from_scratch_reread import run_from_scratch_experiment
from latentrouter.experiments.model_contrastive_reread import build_openclip_patch_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model-contrastive rereading on VL-RouterBench.")
    parser.add_argument("--processed-dir", default="data/processed/vl_routerbench")
    parser.add_argument("--feature-dir", default="data/processed/vl_routerbench/openclip_features")
    parser.add_argument("--patch-cache-dir", default="data/processed/vl_routerbench/openclip_patch49")
    parser.add_argument("--baseline-dir", default="artifacts/vl_routerbench_repro/models")
    parser.add_argument("--output-dir", default="artifacts/model_contrastive_vl_routerbench")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-loss-weight", type=float, default=0.30)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.01)
    parser.add_argument("--reread-bound", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_cache = build_openclip_patch_cache(
        processed_dir=args.processed_dir,
        feature_dir=args.feature_dir,
        cache_dir=args.patch_cache_dir,
        device=args.device,
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for seed in args.seeds:
        print(f"Training joint model from scratch, seed={seed}, reread_bound={args.reread_bound}", flush=True)
        baseline_path = Path(args.baseline_dir) / f"oriented_{seed}.pkl"
        summary = run_from_scratch_experiment(
            processed_dir=args.processed_dir,
            feature_dir=args.feature_dir,
            patch_cache=patch_cache,
            output_dir=output_root / f"seed_{seed}",
            seed=seed,
            baseline_path=baseline_path,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            initial_loss_weight=args.initial_loss_weight,
            attention_entropy_weight=args.attention_entropy_weight,
            reread_bound=args.reread_bound,
            device=args.device,
        )
        summaries.append(summary)
        print(
            f"seed={seed} metric={summary['primary_metric']} "
            f"baseline={summary['baseline_primary_metric']:.6f} "
            f"joint={summary['joint_primary_metric']:.6f} "
            f"delta={summary['delta_vs_baseline']:+.6f}",
            flush=True,
        )
    aggregate = {
        "seed_count": len(summaries),
        "primary_metric": summaries[0]["primary_metric"] if summaries else None,
        "reread_bound": args.reread_bound,
        "mean_baseline_primary_metric": sum(row["baseline_primary_metric"] for row in summaries) / len(summaries),
        "mean_joint_primary_metric": sum(row["joint_primary_metric"] for row in summaries) / len(summaries),
        "mean_delta_vs_baseline": sum(row["delta_vs_baseline"] for row in summaries) / len(summaries),
        "mean_shuffled_delta_vs_joint": sum(row["shuffled_delta_vs_joint"] for row in summaries) / len(summaries),
        "mean_training_time_seconds": sum(row["training_time_seconds"] for row in summaries) / len(summaries),
        "mean_attention_entropy": sum(row["attention"]["attention_entropy"] for row in summaries) / len(summaries),
        "mean_attention_peak": sum(row["attention"]["attention_peak"] for row in summaries) / len(summaries),
        "mean_prediction_time_seconds": sum(row["prediction_time_seconds"] for row in summaries) / len(summaries),
    }
    with (output_root / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": summaries, "aggregate": aggregate}, handle, indent=2, sort_keys=True)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
