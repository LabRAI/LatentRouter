"""SigLIP image embeddings for every VL-RouterBench sample.

Every image referenced by a sample is encoded with the VisionEncoder of the official VL-RouterBench repository
(L2-normalized; samples with several images use the re-normalized mean). Images stored in the VLMEvalKit TSV
files are matched to samples by their `index` column (`id` for MMMU_DEV_VAL).

Usage:
    python scripts/extract_vl_image_features.py --vl-repo VL-RouterBench --root data/vl_routerbench \
        --tsv-dir data/vl_routerbench/vlm_router_data/TSV_images
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TSV_DATASETS = {"DocVQA_VAL", "TextVQA_VAL", "AI2D_TEST", "InfoVQA_VAL", "MMMU_DEV_VAL", "HallusionBench"}


def load_tsv(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = pd.read_csv(path, sep="\t")
    frame["image"] = frame["image"].astype(str)
    images = {str(i): img for i, img in zip(frame["index"], frame["image"])}
    for key, value in list(images.items()):
        if len(value) <= 64 and value in images and len(images[value]) > 64:
            images[key] = images[value]
    return frame, images


def decode(cell: str) -> list[Image.Image]:
    cell = cell.strip()
    blobs = ast.literal_eval(cell) if cell.startswith("[") else [cell]
    return [Image.open(BytesIO(base64.b64decode(b))).convert("RGB") for b in blobs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vl-repo", required=True, help="clone of the official VL-RouterBench repository")
    parser.add_argument("--root", required=True, help="VL-RouterBench dataset directory (BENCHMARKS/, data/, ...)")
    parser.add_argument("--tsv-dir", required=True, help="directory with the VLMEvalKit TSV files")
    parser.add_argument("--encoder", default="google/siglip-large-patch16-384")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sys.path.insert(0, args.vl_repo)
    from routers.features.vision_encoder import VisionEncoder

    root = Path(args.root)
    output = Path(args.output) if args.output else root / "EMBEDDINGS" / "vision" / f"{args.encoder.split('/')[-1]}.parquet"
    samples = [json.loads(line) for f in sorted((root / "BENCHMARKS").glob("*/*_samples.jsonl")) for line in open(f, encoding="utf-8")]

    cells: list[tuple[str, str]] = []
    tables: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}
    positions: dict[tuple[str, str], dict[str, int]] = {}
    for s in samples:
        sid, dataset = s["sample_id"], s["dataset"]
        assets = [a for a in s.get("assets", []) if isinstance(a, dict) and a.get("type") == "image_tsv"]
        if assets:
            for a in assets:
                if a["tsv_file"] not in tables:
                    tables[a["tsv_file"]] = load_tsv(Path(a["tsv_file"]))
                frame, images = tables[a["tsv_file"]]
                row = frame.iloc[int(a["index"])]
                cells.append((sid, images.get(str(row["index"]), row["image"])))
        elif dataset in TSV_DATASETS:
            key = str(Path(args.tsv_dir) / f"{dataset}.tsv")
            if key not in tables:
                tables[key] = load_tsv(Path(key))
            frame, images = tables[key]
            column = "id" if dataset == "MMMU_DEV_VAL" else "index"
            if (key, column) not in positions:
                positions[(key, column)] = {str(v): i for i, v in enumerate(frame[column].tolist())}
            pos = positions[(key, column)].get(sid.split("/", 1)[1])
            if pos is not None:
                row = frame.iloc[pos]
                cells.append((sid, images.get(str(row["index"]), row["image"])))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VisionEncoder(model_name=args.encoder, device=device, batch_size=args.batch_size, normalize=True)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(cells), 512):
        batch, owners = [], []
        for sid, cell in cells[start : start + 512]:
            for img in decode(cell):
                batch.append(img)
                owners.append(sid)
        for sid, vector in zip(owners, encoder._extract_batch_from_images(batch, batch_size=args.batch_size)):
            grouped[sid].append(vector)
        print(f"encoded {min(start + 512, len(cells))}/{len(cells)}", flush=True)

    dim = len(next(iter(grouped.values()))[0])
    rows = []
    for s in samples:
        vectors = grouped.get(s["sample_id"])
        if not vectors:
            emb = np.zeros(dim, dtype=np.float32)
        elif len(vectors) == 1:
            emb = vectors[0]
        else:
            emb = np.mean(np.stack(vectors), axis=0)
            norm = np.linalg.norm(emb)
            emb = emb / norm if norm > 0 else emb
        rows.append({"sample_id": s["sample_id"], "embedding": np.asarray(emb, dtype=np.float32).tolist()})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)


if __name__ == "__main__":
    main()
