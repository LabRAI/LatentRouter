from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentrouter.experiments.model_contrastive_reread import (
    RereadRunConfig,
    build_openclip_patch_cache,
    run_reread_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-contrastive spatial rereading on MMR-Bench.")
    parser.add_argument("--processed-dir", default="data/processed/mmr_official")
    parser.add_argument("--feature-dir", default="data/processed/mmr_official/openclip_features")
    parser.add_argument("--patch-cache-dir", default="data/processed/mmr_official/openclip_patch49")
    parser.add_argument("--baseline-dir", default="artifacts/mmr_repro/models")
    parser.add_argument("--output-dir", default="artifacts/model_contrastive_reread_mmr")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--modes", nargs="+", choices=["contrast", "generic"], default=["contrast", "generic"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--reread-bound", type=float, default=0.10)
    parser.add_argument("--patch-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-patch-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_cache = build_openclip_patch_cache(
        processed_dir=args.processed_dir,
        feature_dir=args.feature_dir,
        cache_dir=args.patch_cache_dir,
        device=args.device,
        batch_size=args.patch_batch_size,
        force=args.force_patch_cache,
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for mode in args.modes:
        for seed in args.seeds:
            baseline_path = Path(args.baseline_dir) / f"oriented_{seed}.pkl"
            if not baseline_path.exists():
                raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_path}")
            config = RereadRunConfig(
                mode=mode,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                inference_batch_size=args.inference_batch_size,
                learning_rate=args.learning_rate,
                entropy_weight=args.entropy_weight,
                reread_bound=args.reread_bound,
                device=args.device,
            )
            run_dir = output_root / f"{mode}_{seed}"
            print(f"Running mode={mode} seed={seed}", flush=True)
            summary = run_reread_experiment(
                baseline_path=baseline_path,
                processed_dir=args.processed_dir,
                feature_dir=args.feature_dir,
                patch_cache=patch_cache,
                output_dir=run_dir,
                seed=seed,
                config=config,
            )
            summaries.append(summary)
            print(
                f"mode={mode} seed={seed} baseline={summary['baseline_nAUC']:.6f} "
                f"reread={summary['reread_nAUC']:.6f} delta={summary['nAUC_delta']:+.6f}",
                flush=True,
            )

    aggregate: dict[str, dict[str, float | int]] = {}
    for mode in args.modes:
        rows = [summary for summary in summaries if summary["mode"] == mode]
        aggregate[mode] = {
            "seed_count": len(rows),
            "mean_baseline_nAUC": sum(row["baseline_nAUC"] for row in rows) / len(rows),
            "mean_reread_nAUC": sum(row["reread_nAUC"] for row in rows) / len(rows),
            "mean_nAUC_delta": sum(row["nAUC_delta"] for row in rows) / len(rows),
            "mean_shuffled_delta_vs_reread": sum(row["shuffled_nAUC_delta_vs_reread"] for row in rows) / len(rows),
            "mean_prediction_slowdown": sum(row["prediction_slowdown"] for row in rows) / len(rows),
            "mean_net_helpful_flip_rate": sum(row["flips"]["net_helpful_flip_rate"] for row in rows) / len(rows),
        }
    payload = {"runs": summaries, "aggregate": aggregate}
    with (output_root / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
