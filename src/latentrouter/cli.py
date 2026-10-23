from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from latentrouter.benchmarks import BENCHMARKS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="latentrouter", description="Train and evaluate LatentRouter on a routing benchmark.")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/vl_routerbench.yaml")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--device", default=None, help="cpu, cuda or auto (overrides the config)")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    if args.device:
        config.setdefault("router", {})["device"] = args.device
    benchmark = BENCHMARKS[config["benchmark"]]
    for seed in args.seed:
        router, result = benchmark.run(config, seed)
        out = Path(args.output_dir) / config["benchmark"] / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        router.save(out / "router.pkl")
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        summary = {k: v for k, v in result.items() if k not in ("trials",)}
        print(json.dumps(summary))
    return 0
