from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from latentrouter.config import FeaturesConfig
from latentrouter.embedding.backends import create_encoder
from latentrouter.io import ensure_dir, read_json, write_json
from latentrouter.schemas import FeatureArtifacts, RouterDatasetBundle


def _decode_image_paths(value: object) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value if str(item)]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item)]
            except json.JSONDecodeError:
                pass
        return [text]
    return [str(value)]


def _load_samples(processed_dir: str | Path) -> pd.DataFrame:
    samples = pd.read_parquet(Path(processed_dir) / "samples.parquet")
    if "image_paths" in samples.columns:
        samples["image_paths"] = samples["image_paths"].apply(_decode_image_paths)
    return samples


def _build_side_features(samples: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[str]]]:
    group_categories = sorted(samples.get("group_id", pd.Series(["unknown"] * len(samples))).fillna("unknown").astype(str).unique().tolist())
    dataset_categories = sorted(samples["dataset_name"].fillna("unknown").astype(str).unique().tolist())
    mode_categories = sorted(samples["mode_id"].fillna("default").astype(str).unique().tolist())

    group_index = {value: idx for idx, value in enumerate(group_categories)}
    dataset_index = {value: idx for idx, value in enumerate(dataset_categories)}
    mode_index = {value: idx for idx, value in enumerate(mode_categories)}

    group_one_hot = np.zeros((len(samples), len(group_categories)), dtype=np.float32)
    dataset_one_hot = np.zeros((len(samples), len(dataset_categories)), dtype=np.float32)
    mode_one_hot = np.zeros((len(samples), len(mode_categories)), dtype=np.float32)

    for row_idx, value in enumerate(samples.get("group_id", pd.Series(["unknown"] * len(samples))).fillna("unknown").astype(str)):
        group_one_hot[row_idx, group_index[value]] = 1.0
    for row_idx, value in enumerate(samples["dataset_name"].fillna("unknown").astype(str)):
        dataset_one_hot[row_idx, dataset_index[value]] = 1.0
    for row_idx, value in enumerate(samples["mode_id"].fillna("default").astype(str)):
        mode_one_hot[row_idx, mode_index[value]] = 1.0

    question_length = samples["question"].fillna("").astype(str).str.len().to_numpy(dtype=np.float32).reshape(-1, 1)
    image_count = samples["image_count"].fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)

    question_scale = max(float(question_length.max(initial=1.0)), 1.0)
    image_scale = max(float(image_count.max(initial=1.0)), 1.0)
    numeric = np.concatenate([question_length / question_scale, image_count / image_scale], axis=1)
    features = np.concatenate([group_one_hot, dataset_one_hot, mode_one_hot, numeric], axis=1)
    manifest = {
        "group_categories": group_categories,
        "dataset_categories": dataset_categories,
        "mode_categories": mode_categories,
    }
    return features.astype(np.float32), manifest


def build_feature_store(processed_dir: str | Path, config: FeaturesConfig) -> FeatureArtifacts:
    processed_dir = Path(processed_dir)
    samples = _load_samples(processed_dir)
    feature_dir = ensure_dir(config.output_dir or processed_dir / "features")

    sample_index_path = feature_dir / "sample_index.parquet"
    text_embeddings_path = feature_dir / "text_embeddings.npy"
    image_embeddings_path = feature_dir / "image_embeddings.npy"
    side_features_path = feature_dir / "side_features.npy"
    fused_features_path = feature_dir / "fused_features.npy"
    manifest_path = feature_dir / "manifest.json"

    if manifest_path.exists() and not config.force:
        return FeatureArtifacts(
            feature_dir=feature_dir,
            sample_index_path=sample_index_path,
            text_embeddings_path=text_embeddings_path,
            image_embeddings_path=image_embeddings_path,
            side_features_path=side_features_path,
            fused_features_path=fused_features_path,
            manifest_path=manifest_path,
        )

    encoder = create_encoder(
        backend=config.backend,
        hashing_dim=config.hashing_dim,
        openclip_model=config.openclip_model,
        openclip_pretrained=config.openclip_pretrained,
        coconut_repo_path=config.coconut_repo_path,
        coconut_model_id=config.coconut_model_id,
        coconut_checkpoint=config.coconut_checkpoint,
        coconut_checkpoint_filename=config.coconut_checkpoint_filename,
        coconut_latent_tokens=config.coconut_latent_tokens,
        coconut_max_length=config.coconut_max_length,
        batch_size=config.batch_size,
        device=config.device,
    )

    text_embeddings = encoder.encode_texts(samples["question"].fillna("").astype(str).tolist())
    image_embeddings = encoder.encode_image_sets(samples["image_paths"].tolist())
    side_features, side_manifest = _build_side_features(samples)
    fused_features = np.concatenate([text_embeddings, image_embeddings, side_features], axis=1).astype(np.float32)

    sample_index_columns = [column for column in ["sample_id", "benchmark_id", "group_id", "dataset_name", "mode_id"] if column in samples.columns]
    samples[sample_index_columns].to_parquet(sample_index_path, index=False)
    np.save(text_embeddings_path, text_embeddings)
    np.save(image_embeddings_path, image_embeddings)
    np.save(side_features_path, side_features)
    np.save(fused_features_path, fused_features)
    write_json(
        manifest_path,
        {
            "backend": config.backend,
            "openclip_model": config.openclip_model,
            "openclip_pretrained": config.openclip_pretrained,
            "coconut_repo_path": config.coconut_repo_path,
            "coconut_model_id": config.coconut_model_id,
            "coconut_checkpoint": config.coconut_checkpoint,
            "coconut_checkpoint_filename": config.coconut_checkpoint_filename,
            "coconut_latent_tokens": int(config.coconut_latent_tokens),
            "coconut_max_length": int(config.coconut_max_length),
            "sample_count": int(len(samples)),
            "text_dim": int(text_embeddings.shape[1]),
            "image_dim": int(image_embeddings.shape[1]),
            "side_dim": int(side_features.shape[1]),
            "fused_dim": int(fused_features.shape[1]),
            **side_manifest,
        },
    )
    return FeatureArtifacts(
        feature_dir=feature_dir,
        sample_index_path=sample_index_path,
        text_embeddings_path=text_embeddings_path,
        image_embeddings_path=image_embeddings_path,
        side_features_path=side_features_path,
        fused_features_path=fused_features_path,
        manifest_path=manifest_path,
    )


def load_router_bundle(
    processed_dir: str | Path,
    split: str | None = None,
    feature_dir: str | Path | None = None,
) -> RouterDatasetBundle:
    processed_dir = Path(processed_dir)
    feature_dir = Path(feature_dir) if feature_dir is not None else processed_dir / "features"
    samples = _load_samples(processed_dir)
    splits_path = processed_dir / "splits.parquet"
    if splits_path.exists():
        splits = pd.read_parquet(splits_path)
        if "split" in samples.columns:
            samples = samples.drop(columns=["split"])
        samples = samples.merge(splits, on="sample_id", how="left")
    else:
        samples["split"] = "train"

    if split is not None:
        samples = samples.loc[samples["split"] == split].reset_index(drop=True)
    else:
        samples = samples.reset_index(drop=True)

    sample_ids = samples["sample_id"].astype(str).to_numpy()
    fused_features = np.load(feature_dir / "fused_features.npy")
    sample_index = pd.read_parquet(feature_dir / "sample_index.parquet")
    feature_manifest = read_json(feature_dir / "manifest.json") if (feature_dir / "manifest.json").exists() else {}
    order = sample_index["sample_id"].astype(str).tolist()
    index_lookup = {sample_id: idx for idx, sample_id in enumerate(order)}
    feature_rows = [index_lookup[sample_id] for sample_id in sample_ids]
    feature_matrix = fused_features[feature_rows]

    outcomes = pd.read_parquet(processed_dir / "outcomes.parquet")
    manifest = read_json(processed_dir / "manifest.json")
    model_ids = [item["model_id"] for item in read_json(processed_dir / "models.json")]

    correctness_frame = outcomes.pivot(index="sample_id", columns="model_id", values="quality")
    cost_frame = outcomes.pivot(index="sample_id", columns="model_id", values="cost")
    token_frame = outcomes.pivot(index="sample_id", columns="model_id", values="token_count")

    correctness = correctness_frame.reindex(index=sample_ids, columns=model_ids).to_numpy(dtype=float)
    costs = cost_frame.reindex(index=sample_ids, columns=model_ids).to_numpy(dtype=float)
    token_counts = token_frame.reindex(index=sample_ids, columns=model_ids).to_numpy(dtype=float)
    availability = ~np.isnan(correctness) & ~np.isnan(costs)

    return RouterDatasetBundle(
        sample_ids=sample_ids,
        model_ids=model_ids,
        features=feature_matrix,
        correctness=correctness,
        costs=costs,
        token_counts=token_counts,
        availability=availability,
        sample_frame=samples.reset_index(drop=True),
        benchmark_id=str(manifest.get("benchmark_id", "mmr")),
        protocol_id=str(manifest.get("protocol_id", "mmr_cost_quality")),
        evaluation_metadata={
            "cost_bounds": manifest.get("cost_bounds", {}),
            "manifest": manifest,
            "feature_manifest": feature_manifest,
        },
    )
