from __future__ import annotations

import json
import math
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latentrouter.benchmarks.base import BenchmarkAdapter, BenchmarkProtocol
from latentrouter.benchmarks.common import (
    lambda_grid,
    materialize_routes,
    oracle_frontier,
    single_model_frontier,
    write_evaluation_outputs,
)
from latentrouter.config import EvaluationConfig
from latentrouter.data.splits import create_splits
from latentrouter.evaluation.metrics import best_single_model, compute_nauc, compute_ps, compute_qnc
from latentrouter.io import ensure_dir, stable_hash, write_json
from latentrouter.schemas import NormalizedArtifacts, RouterDatasetBundle

MODEL_SUFFIXES = [
    ("_completion_tokens", "completion_tokens"),
    ("_prompt_tokens", "prompt_tokens"),
    ("_is_correct", "is_correct"),
    ("_prediction", "prediction"),
    ("_response", "prediction"),
    ("_raw_output", "raw_output"),
    ("_correct", "is_correct"),
    ("_tokens", "token_count"),
    ("_token", "token_count"),
    ("_score", "score"),
    ("_cost", "cost"),
    ("_pred", "prediction"),
]

PRIMARY_SAMPLE_COLUMNS = {
    "sample_id",
    "id",
    "dataset_idx",
    "dataset_name",
    "mode_id",
    "mode",
    "question",
    "query",
    "instruction",
    "prompt",
    "answer",
    "ground_truth",
    "label",
    "gt",
    "image_paths",
    "image_path",
    "images",
    "image",
    "img_path",
}

MMR_SNAPSHOT_ALLOW_PATTERNS = [
    "MMR_Bench.csv",
    "MMR-Bench.csv",
    "images.tar.gz",
    "MathVerse/*",
    "MathVision/*",
    "MathVista/*",
    "OCRBench/*",
    "RealWorldQA/*",
    "MMStar/*",
    "SEEDBenchv2Plus/*",
]

MMR_PREFIX_TO_DIR = {
    "MathVerse": "MathVerse",
    "MathVision": "MathVision",
    "MathVista": "MathVista",
    "OCRBench": "OCRBench",
    "RealWorldQA": "RealWorldQA",
    "MMStar": "MMStar",
    "SEEDBench2_Plus": "SEEDBenchv2Plus",
}


def _normalize_scalar(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, dict, str, bool, int, float)):
        return value
    return str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _coerce_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (np.integer, int, np.floating, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        lowered = text.lower()
        if lowered in {"true", "yes", "correct"}:
            return 1.0
        if lowered in {"false", "no", "incorrect"}:
            return 0.0
        return np.nan


def _coerce_correctness(value: Any) -> float:
    score = _coerce_float(value)
    if np.isnan(score):
        return np.nan
    return float(score > 0.5) if score not in {0.0, 1.0} else score


def _parse_image_paths(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        if text.startswith("["):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, list):
                    return [str(item) for item in decoded if str(item)]
            except json.JSONDecodeError:
                pass
        if "|" in text:
            return [part for part in text.split("|") if part]
        return [text]
    return [str(value)]


def _choose_dataset_idx_prefix(dataset_idx: str) -> str | None:
    text = str(dataset_idx)
    best = None
    for prefix in MMR_PREFIX_TO_DIR:
        if text == prefix or text.startswith(prefix + "_"):
            if best is None or len(prefix) > len(best):
                best = prefix
    return best


def _infer_snapshot_image_path(data_root: Path, dataset_idx: str) -> str:
    prefix = _choose_dataset_idx_prefix(dataset_idx)
    if prefix is None:
        return ""
    folder = MMR_PREFIX_TO_DIR[prefix]
    remainder = str(dataset_idx)
    if remainder.startswith(prefix + "_"):
        remainder = remainder[len(prefix) + 1 :]
    else:
        remainder = remainder.split("_")[-1]
    base_dir = (data_root / folder).resolve()
    for suffix in (".jpg", ".png", ".jpeg", ".webp"):
        candidate = base_dir / f"{remainder}{suffix}"
        if candidate.exists():
            return str(candidate)
    fallback = base_dir / f"{remainder}.jpg"
    return str(fallback) if fallback.exists() else ""


def _attach_snapshot_image_paths(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    enriched = frame.copy()
    if "image_paths" in enriched.columns and enriched["image_paths"].notna().any():
        return enriched
    if "img_path" in enriched.columns and enriched["img_path"].notna().any():
        enriched["image_paths"] = enriched["img_path"].apply(_parse_image_paths)
        return enriched
    if "dataset_idx" not in enriched.columns:
        enriched["image_paths"] = [[] for _ in range(len(enriched))]
        return enriched
    enriched["image_paths"] = [
        ([path] if path else [])
        for path in (_infer_snapshot_image_path(data_root, dataset_idx) for dataset_idx in enriched["dataset_idx"].tolist())
    ]
    return enriched


def _load_local_mmr_layout(layout_root: Path) -> pd.DataFrame:
    merged_candidates = [layout_root / "MMR_Bench.csv", layout_root / "MMR-Bench.csv"]
    merged_path = next((candidate for candidate in merged_candidates if candidate.exists()), None)
    if merged_path is None:
        raise FileNotFoundError(f"Expected MMR_Bench.csv or MMR-Bench.csv under {layout_root}")
    tar_path = layout_root / "images.tar.gz"
    needs_extract = tar_path.exists() and not any((layout_root / folder).exists() for folder in set(MMR_PREFIX_TO_DIR.values()))
    if needs_extract:
        with tarfile.open(tar_path, "r:gz") as archive:
            archive.extractall(layout_root)
    frame = pd.read_csv(merged_path)
    frame = _attach_snapshot_image_paths(frame, layout_root)
    frame.attrs["source_layout"] = "local_mmr_layout"
    frame.attrs["image_root"] = str(layout_root.resolve())
    return frame


def _load_hf_snapshot_frame(repo_id: str, adapter_options: dict[str, Any] | None) -> pd.DataFrame:
    adapter_options = adapter_options or {}
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Rich MMR snapshot loading requires huggingface-hub. "
            "Install latentrouter with the '[hf]' extra."
        ) from exc

    revision = adapter_options.get("hf_snapshot_revision")
    local_dir_option = adapter_options.get("hf_snapshot_local_dir")
    snapshot_force = bool(adapter_options.get("hf_snapshot_force", False))
    allow_patterns = adapter_options.get("hf_snapshot_allow_patterns", MMR_SNAPSHOT_ALLOW_PATTERNS)
    local_dir = Path(local_dir_option).expanduser() if local_dir_option else None

    if local_dir is not None and local_dir.exists() and snapshot_force:
        shutil.rmtree(local_dir)
    if local_dir is not None and (local_dir / "MMR_Bench.csv").exists() and not snapshot_force:
        snapshot_root = local_dir
    else:
        download_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "revision": revision,
            "allow_patterns": allow_patterns,
        }
        if local_dir is not None:
            local_dir.mkdir(parents=True, exist_ok=True)
            download_kwargs["local_dir"] = str(local_dir)
            download_kwargs["local_dir_use_symlinks"] = False
        snapshot_root = Path(snapshot_download(**download_kwargs))

    frame = _load_local_mmr_layout(snapshot_root)
    frame.attrs["source_layout"] = "hf_snapshot_assets"
    frame.attrs["hf_snapshot_repo"] = repo_id
    return frame


def parse_dataset_fields(dataset_idx: str) -> tuple[str, str]:
    text = (dataset_idx or "unknown").strip()
    patterns = ["::", "__", "|", "/", ":"]
    for pattern in patterns:
        if pattern in text:
            left, right = text.split(pattern, 1)
            if left and right:
                return left, right
    match = re.match(r"^(?P<dataset>[A-Za-z0-9._-]+)[\[\(](?P<mode>[^\]\)]+)[\]\)]$", text)
    if match:
        return match.group("dataset"), match.group("mode")
    suffix_match = re.match(r"^(?P<dataset>.+)_(?P<row_id>\d+)$", text)
    if suffix_match:
        return suffix_match.group("dataset"), "default"
    return text or "unknown", "default"


def discover_model_groups(columns: list[str]) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for column in columns:
        for suffix, canonical_name in MODEL_SUFFIXES:
            if column.endswith(suffix):
                model_id = column[: -len(suffix)]
                if model_id:
                    groups.setdefault(model_id, {})[canonical_name] = column
                break
    filtered = {
        model_id: fields
        for model_id, fields in groups.items()
        if {"prediction", "is_correct", "cost"} & set(fields)
    }
    if not filtered:
        raise ValueError("No model output columns were discovered in the source table.")
    return dict(sorted(filtered.items()))


def load_frame(source: str, split: str = "train", adapter_options: dict[str, Any] | None = None) -> pd.DataFrame:
    if source.startswith("hf://"):
        dataset_name = source.removeprefix("hf://")
        prefer_snapshot_assets = bool((adapter_options or {}).get("prefer_snapshot_assets", dataset_name == "gh0stHunter/MMR-Bench"))
        if prefer_snapshot_assets and dataset_name == "gh0stHunter/MMR-Bench":
            try:
                return _load_hf_snapshot_frame(dataset_name, adapter_options)
            except Exception:
                if bool((adapter_options or {}).get("strict_snapshot_assets", False)):
                    raise
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face source requested but 'datasets' is not installed. "
                "Install latentrouter with the '[hf]' extra."
            ) from exc
        dataset = load_dataset(dataset_name, split=split)
        frame = dataset.to_pandas()
        frame.attrs["source_layout"] = "hf_datasets_table"
        return frame

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if source_path.is_dir():
        if (source_path / "MMR_Bench.csv").exists():
            return _load_local_mmr_layout(source_path)
        raise FileNotFoundError(f"Directory source is not a recognized MMR layout: {source}")
    if source_path.suffix == ".parquet":
        frame = pd.read_parquet(source_path)
        frame.attrs["source_layout"] = "file_parquet"
        return frame
    if source_path.suffix == ".jsonl":
        frame = pd.read_json(source_path, lines=True)
        frame.attrs["source_layout"] = "file_jsonl"
        return frame
    if source_path.suffix == ".json":
        frame = pd.read_json(source_path)
        frame.attrs["source_layout"] = "file_json"
        return frame
    frame = pd.read_csv(source_path)
    frame.attrs["source_layout"] = "file_csv"
    return frame


def normalize_source(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    model_groups = discover_model_groups(list(frame.columns))
    model_columns = {column for fields in model_groups.values() for column in fields.values()}

    sample_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    model_registry = [
        {"model_id": model_id, "fields": sorted(fields.keys())}
        for model_id, fields in model_groups.items()
    ]

    for row_idx, row in frame.reset_index(drop=True).iterrows():
        raw = row.to_dict()
        dataset_idx = str(
            raw.get("dataset_idx")
            or raw.get("dataset_name")
            or raw.get("dataset")
            or "unknown"
        )
        dataset_name, derived_mode = parse_dataset_fields(dataset_idx)
        mode_id = str(raw.get("mode_id") or raw.get("mode") or derived_mode or "default")
        question = str(
            raw.get("question")
            or raw.get("query")
            or raw.get("instruction")
            or raw.get("prompt")
            or ""
        )
        answer = str(raw.get("answer") or raw.get("ground_truth") or raw.get("label") or raw.get("gt") or "")
        image_paths = _parse_image_paths(
            raw.get("image_paths") or raw.get("images") or raw.get("image_path") or raw.get("image") or raw.get("img_path")
        )
        sample_id = str(raw.get("sample_id") or raw.get("id") or stable_hash(f"{dataset_idx}::{row_idx}::{question}"))
        metadata = {
            key: _normalize_scalar(value)
            for key, value in raw.items()
            if key not in model_columns and key not in PRIMARY_SAMPLE_COLUMNS and not _is_missing(value)
        }

        sample_rows.append(
            {
                "sample_id": sample_id,
                "benchmark_id": "mmr",
                "dataset_idx": dataset_idx,
                "group_id": raw.get("group_id") or raw.get("task_group") or raw.get("family") or dataset_name,
                "dataset_name": raw.get("dataset_name") or dataset_name,
                "mode_id": mode_id,
                "question": question,
                "answer": answer,
                "image_paths": image_paths,
                "image_count": len(image_paths),
                "metadata_json": json.dumps(metadata, sort_keys=True),
            }
        )

        for model_id, fields in model_groups.items():
            prediction = raw.get(fields.get("prediction", ""))
            quality = _coerce_correctness(raw.get(fields.get("is_correct")))
            cost = _coerce_float(raw.get(fields.get("cost")))
            token_count = _coerce_float(raw.get(fields.get("token_count")))
            prompt_tokens = _coerce_float(raw.get(fields.get("prompt_tokens")))
            completion_tokens = _coerce_float(raw.get(fields.get("completion_tokens")))
            raw_output = raw.get(fields.get("raw_output"))

            if prediction is None and np.isnan(quality) and np.isnan(cost) and np.isnan(token_count):
                continue

            outcome_rows.append(
                {
                    "sample_id": sample_id,
                    "model_id": model_id,
                    "quality": quality,
                    "prediction": "" if prediction is None else str(prediction),
                    "is_correct": quality,
                    "token_count": token_count,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": cost,
                    "raw_output": None if raw_output is None else str(raw_output),
                    "metadata_json": json.dumps({}, sort_keys=True),
                }
            )

    samples = pd.DataFrame(sample_rows).sort_values("sample_id").reset_index(drop=True)
    outcomes = pd.DataFrame(outcome_rows).sort_values(["sample_id", "model_id"]).reset_index(drop=True)
    return samples, outcomes, model_registry


class MMRAdapter(BenchmarkAdapter):
    benchmark_id = "mmr"
    protocol_id = "mmr_cost_quality"

    def prepare(
        self,
        source: str,
        processed_dir: str | Path,
        source_split: str = "train",
        split_manifest: str | Path | None = None,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.2,
        seed: int = 20260308,
        adapter_options: dict[str, Any] | None = None,
    ) -> NormalizedArtifacts:
        processed_dir = ensure_dir(processed_dir)
        frame = load_frame(source, split=source_split, adapter_options=adapter_options)
        samples, outcomes, model_registry = normalize_source(frame)
        splits = create_splits(
            samples,
            split_manifest=split_manifest,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            group_columns=["dataset_name", "mode_id"],
        )

        samples_path = processed_dir / "samples.parquet"
        outcomes_path = processed_dir / "outcomes.parquet"
        models_path = processed_dir / "models.json"
        manifest_path = processed_dir / "manifest.json"
        splits_path = processed_dir / "splits.parquet"

        samples.to_parquet(samples_path, index=False)
        outcomes.to_parquet(outcomes_path, index=False)
        splits.to_parquet(splits_path, index=False)
        write_json(models_path, model_registry)
        write_json(
            manifest_path,
            {
                "adapter_version": 2,
                "benchmark_id": self.benchmark_id,
                "protocol_id": self.protocol_id,
                "source": source,
                "source_split": source_split,
                "split_manifest": None if split_manifest is None else str(split_manifest),
                "split_policy": "manifest" if split_manifest else "generated",
                "sample_count": int(len(samples)),
                "outcome_count": int(len(outcomes)),
                "model_count": int(len(model_registry)),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_layout": frame.attrs.get("source_layout", "unknown"),
                "image_root": frame.attrs.get("image_root"),
                "image_nonempty_count": int((samples["image_count"] > 0).sum()),
                "adapter_options": adapter_options or {},
            },
        )
        return NormalizedArtifacts(
            processed_dir=processed_dir,
            samples_path=samples_path,
            outcomes_path=outcomes_path,
            models_path=models_path,
            manifest_path=manifest_path,
            splits_path=splits_path,
        )


def _slice_metrics(routes: pd.DataFrame, aggregate_by: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in aggregate_by:
        if field not in routes.columns:
            continue
        for value, frame in routes.groupby(field, dropna=False):
            frontier = (
                frame.groupby("lambda_value", as_index=False)
                .agg(mean_cost=("cost", "mean"), mean_quality=("correctness", "mean"))
                .sort_values("mean_cost")
                .reset_index(drop=True)
            )
            single_model = (
                frame.groupby("model_id", as_index=False)
                .agg(cost=("cost", "mean"), quality=("correctness", "mean"))
            )
            ps_value = compute_ps(frontier)
            best_single = best_single_model(single_model)
            rows.append(
                {
                    "slice_field": field,
                    "slice_value": value,
                    "primary_metric_value": compute_nauc(frontier),
                    "nAUC": compute_nauc(frontier),
                    "Ps": ps_value,
                    "QNC": compute_qnc(frontier, single_model),
                    "QNC_99pct": compute_qnc(frontier, single_model, quality_fraction=0.99),
                    "QNC_eps_0_01": compute_qnc(frontier, single_model, quality_epsilon=0.01),
                    "reaches_best_single_quality": bool(ps_value >= float(best_single["quality"])),
                    "sample_count": int(frame["sample_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


class MMRProtocol(BenchmarkProtocol):
    benchmark_id = "mmr"
    protocol_id = "mmr_cost_quality"

    def evaluate(
        self,
        router_name: str,
        predicted_utilities: np.ndarray,
        bundle: RouterDatasetBundle,
        config: EvaluationConfig,
        run_dir: str | Path,
        prediction_wall_time_seconds: float | None = None,
    ) -> dict[str, Any]:
        del prediction_wall_time_seconds
        lambdas = lambda_grid(config)
        routes, frontier = materialize_routes(predicted_utilities, bundle, lambdas)
        single_model = single_model_frontier(bundle)
        oracle = oracle_frontier(bundle, lambdas)
        slice_metrics = _slice_metrics(routes.rename(columns={"selected_model": "model_id"}), config.aggregate_by)
        best_single = best_single_model(single_model)
        ps_value = compute_ps(frontier)

        metrics = {
            "benchmark_id": self.benchmark_id,
            "protocol_id": self.protocol_id,
            "router_name": router_name,
            "primary_metric": "nAUC",
            "primary_metric_value": compute_nauc(frontier),
            "summary_metric_order": [
                "nAUC",
                "Ps",
                "QNC",
                "QNC_99pct",
                "QNC_eps_0_01",
                "reaches_best_single_quality",
                "quality_gap_to_best_single",
                "best_single_model",
                "best_single_quality",
                "best_single_cost",
            ],
            "slice_metric_order": ["nAUC", "Ps", "QNC", "QNC_99pct", "QNC_eps_0_01"],
            "nAUC": compute_nauc(frontier),
            "Ps": ps_value,
            "QNC": compute_qnc(frontier, single_model),
            "QNC_99pct": compute_qnc(frontier, single_model, quality_fraction=0.99),
            "QNC_eps_0_01": compute_qnc(frontier, single_model, quality_epsilon=0.01),
            "best_single_model": best_single["model_id"],
            "best_single_quality": float(best_single["quality"]),
            "best_single_cost": float(best_single["cost"]),
            "reaches_best_single_quality": bool(ps_value >= float(best_single["quality"])),
            "quality_gap_to_best_single": float(ps_value - float(best_single["quality"])),
            "lambda_count": int(len(lambdas)),
            "sample_count": int(len(bundle.sample_ids)),
            "model_count": int(len(bundle.model_ids)),
            "plot_x_label": "Mean cost",
            "plot_y_label": "Mean quality",
            "plot_title": "Cost-quality frontier",
        }
        return write_evaluation_outputs(run_dir, metrics, frontier, routes, single_model, oracle, slice_metrics)
