"""OpenCLIP question and image embeddings for every MMR-Bench query.

Usage:
    python scripts/extract_mmr_features.py --root data/mmr_bench --output data/mmr_bench/clip_vit_b32_openai.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from latentrouter.benchmarks.mmr_bench import image_path, table_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="MMR-Bench directory with the outcome table and the extracted image folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.root)
    table = pd.read_csv(table_path(root))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval().to(device)

    questions = table["question"].astype(str).tolist()
    paths = [image_path(root, idx) for idx in table["dataset_idx"]]
    text, image = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size), desc="text"):
            feats = model.encode_text(tokenizer(questions[i : i + args.batch_size]).to(device))
            text.append((feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy())
        for i in tqdm(range(0, len(paths), args.batch_size), desc="image"):
            batch = []
            for p in paths[i : i + args.batch_size]:
                try:
                    batch.append(preprocess(Image.open(p).convert("RGB")))
                except Exception:
                    batch.append(torch.zeros(3, 224, 224))
            feats = model.encode_image(torch.stack(batch).to(device))
            image.append((feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dataset_idx=table["dataset_idx"].astype(str).to_numpy(),
        text=np.vstack(text).astype(np.float32),
        image=np.vstack(image).astype(np.float32),
    )


if __name__ == "__main__":
    main()
