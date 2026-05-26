from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from latentrouter.config import FeaturesConfig, load_run_config
from latentrouter.data.normalize import prepare_benchmark
from latentrouter.embedding.store import build_feature_store, load_router_bundle
from latentrouter.evaluation.runner import run_router_on_benchmark
from latentrouter.reporting import generate_report
from latentrouter.routers import BaseRouter, create_router


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _build_run_dir(output_dir: str, router_name: str, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{router_name}_{timestamp}"


def _benchmark_arg(args: argparse.Namespace, config) -> str:
    return str(getattr(args, "benchmark", None) or config.data.benchmark)


def _config_arg(args: argparse.Namespace) -> str:
    return str(args.config or _default_config_path())


def cmd_prepare(args: argparse.Namespace) -> int:
    config = load_run_config(_config_arg(args))
    prepare_benchmark(
        benchmark=_benchmark_arg(args, config),
        source=args.source or config.data.source,
        processed_dir=args.processed_dir or config.data.processed_dir,
        source_split=args.source_split or config.data.source_split,
        split_manifest=args.split_manifest or config.data.split_manifest,
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        seed=config.data.seed,
        adapter_options=dict(config.data.adapter_options),
    )
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    config = load_run_config(_config_arg(args))
    feature_config = FeaturesConfig(**asdict(config.features))
    if args.backend:
        feature_config.backend = args.backend
    if args.force:
        feature_config.force = True
    build_feature_store(args.processed_dir or config.data.processed_dir, feature_config)
    return 0


def _fit_router(args: argparse.Namespace) -> tuple[BaseRouter, Path]:
    config = load_run_config(_config_arg(args))
    processed_dir = Path(args.processed_dir or config.data.processed_dir)
    feature_config = FeaturesConfig(**asdict(config.features))
    if getattr(args, "backend", None):
        feature_config.backend = args.backend
    build_feature_store(processed_dir, feature_config)
    feature_dir = feature_config.output_dir or processed_dir / "features"
    train_bundle = load_router_bundle(processed_dir, split="train", feature_dir=feature_dir)
    valid_bundle = load_router_bundle(processed_dir, split="val", feature_dir=feature_dir)
    router_name = args.router or config.router.name
    router = create_router(
        router_name,
        model_ids=train_bundle.model_ids,
        random_seed=config.router.random_seed,
        **dict(config.router.hyperparameters),
    )
    router.fit(train_bundle, valid_bundle)
    model_path = Path(args.model_path or config.router.model_path or f"artifacts/models/{router_name}.pkl")
    router.save(model_path)
    return router, model_path


def cmd_train(args: argparse.Namespace) -> int:
    _fit_router(args)
    return 0


def _load_or_train_router(args: argparse.Namespace) -> tuple[BaseRouter, Path]:
    config = load_run_config(_config_arg(args))
    router_name = args.router or config.router.name
    model_path = Path(args.model_path or config.router.model_path or f"artifacts/models/{router_name}.pkl")
    if model_path.exists() and not args.force_retrain:
        return BaseRouter.load(model_path), model_path
    return _fit_router(args)


def cmd_eval(args: argparse.Namespace) -> int:
    config = load_run_config(_config_arg(args))
    processed_dir = Path(args.processed_dir or config.data.processed_dir)
    router_name = args.router or config.router.name
    router, _ = _load_or_train_router(args)
    feature_config = FeaturesConfig(**asdict(config.features))
    if getattr(args, "backend", None):
        feature_config.backend = args.backend
    build_feature_store(processed_dir, feature_config)
    feature_dir = feature_config.output_dir or processed_dir / "features"
    run_dir = _build_run_dir(config.evaluation.output_dir, router_name, args.run_dir)
    run_router_on_benchmark(
        router_name=router_name,
        router=router,
        processed_dir=processed_dir,
        config=config.evaluation,
        run_dir=run_dir,
        split="test",
        feature_dir=feature_dir,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = load_run_config(_config_arg(args))
    generate_report(args.run_dir, config.evaluation.aggregate_by)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_run_config(_config_arg(args))
    source = args.source or config.data.source
    processed_dir = Path(args.processed_dir or config.data.processed_dir)
    prepare_benchmark(
        benchmark=_benchmark_arg(args, config),
        source=source,
        processed_dir=processed_dir,
        source_split=args.source_split or config.data.source_split,
        split_manifest=args.split_manifest or config.data.split_manifest,
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        seed=config.data.seed,
        adapter_options=dict(config.data.adapter_options),
    )

    feature_config = FeaturesConfig(**asdict(config.features))
    if args.backend:
        feature_config.backend = args.backend
    feature_config.force = args.force
    build_feature_store(processed_dir, feature_config)
    router, _ = _load_or_train_router(args)
    feature_dir = feature_config.output_dir or processed_dir / "features"
    run_dir = _build_run_dir(config.evaluation.output_dir, args.router or config.router.name, args.run_dir)
    evaluation = run_router_on_benchmark(
        router_name=args.router or config.router.name,
        router=router,
        processed_dir=processed_dir,
        config=config.evaluation,
        run_dir=run_dir,
        split="test",
        feature_dir=feature_dir,
    )
    generate_report(run_dir, config.evaluation.aggregate_by)
    metric = evaluation["metrics"]["primary_metric"]
    value = evaluation["metrics"]["primary_metric_value"]
    print(f"{metric}: {value:.6f}")
    print(f"run_dir: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatentRouter offline multimodal router benchmark harness")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Normalize raw benchmark data.")
    prepare.add_argument("--benchmark", default=None, choices=["mmr", "vl_routerbench"])
    prepare.add_argument("--source", default=None)
    prepare.add_argument("--source-split", default=None)
    prepare.add_argument("--processed-dir", default=None)
    prepare.add_argument("--split-manifest", default=None)
    prepare.set_defaults(func=cmd_prepare)

    embed = subparsers.add_parser("embed", help="Build cached query features.")
    embed.add_argument("--processed-dir", default=None)
    embed.add_argument("--backend", default=None, choices=["hashing", "openclip", "openclip_coconut"])
    embed.add_argument("--force", action="store_true")
    embed.set_defaults(func=cmd_embed)

    train = subparsers.add_parser("train", help="Train a router.")
    train.add_argument("--benchmark", default=None, choices=["mmr", "vl_routerbench"])
    train.add_argument("--processed-dir", default=None)
    train.add_argument("--backend", default=None, choices=["hashing", "openclip", "openclip_coconut"])
    train.add_argument("--router", default=None)
    train.add_argument("--model-path", default=None)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("eval", help="Evaluate a router on the test split.")
    evaluate.add_argument("--benchmark", default=None, choices=["mmr", "vl_routerbench"])
    evaluate.add_argument("--processed-dir", default=None)
    evaluate.add_argument("--backend", default=None, choices=["hashing", "openclip", "openclip_coconut"])
    evaluate.add_argument("--router", default=None)
    evaluate.add_argument("--model-path", default=None)
    evaluate.add_argument("--run-dir", default=None)
    evaluate.add_argument("--force-retrain", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    report = subparsers.add_parser("report", help="Regenerate plots and summary for an existing run.")
    report.add_argument("--run-dir", required=True)
    report.set_defaults(func=cmd_report)

    run = subparsers.add_parser("run", help="End-to-end prepare, embed, train, eval, and report.")
    run.add_argument("--benchmark", default=None, choices=["mmr", "vl_routerbench"])
    run.add_argument("--source", default=None)
    run.add_argument("--source-split", default=None)
    run.add_argument("--processed-dir", default=None)
    run.add_argument("--split-manifest", default=None)
    run.add_argument("--backend", default=None, choices=["hashing", "openclip", "openclip_coconut"])
    run.add_argument("--router", default=None)
    run.add_argument("--model-path", default=None)
    run.add_argument("--run-dir", default=None)
    run.add_argument("--force", action="store_true")
    run.add_argument("--force-retrain", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
