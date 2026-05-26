#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_SPECS = {
    "cheap_small_3b": {"cost": 0.20, "tokens": 80},
    "ocr_specialist_7b": {"cost": 0.55, "tokens": 140},
    "chart_specialist_7b": {"cost": 0.60, "tokens": 150},
    "reasoning_13b": {"cost": 0.85, "tokens": 190},
    "generalist_30b": {"cost": 1.20, "tokens": 260},
}

GROUPS = [
    ("ocr_document", "ocr", "read dense receipt text and document snippets"),
    ("chart_diagram", "chart", "interpret chart bars axes and diagram layout"),
    ("math_reasoning", "reasoning", "solve visual math and multi step reasoning"),
    ("general_vqa", "general", "answer everyday visual question"),
]


def _quality(model_id: str, group_id: str, difficulty: str, rng: np.random.Generator) -> int:
    easy = difficulty == "easy"
    medium = difficulty == "medium"
    hard = difficulty == "hard"
    base = 0.20 + 0.25 * easy + 0.10 * medium
    if model_id == "cheap_small_3b":
        prob = base + (0.25 if group_id == "general" and not hard else 0.0)
    elif model_id == "generalist_30b":
        prob = 0.72 + 0.12 * easy - 0.07 * hard
    elif model_id == "ocr_specialist_7b":
        prob = 0.82 if group_id == "ocr_document" else 0.45 + 0.12 * easy
    elif model_id == "chart_specialist_7b":
        prob = 0.82 if group_id == "chart_diagram" else 0.43 + 0.12 * easy
    elif model_id == "reasoning_13b":
        prob = 0.82 if group_id == "math_reasoning" else 0.46 + 0.10 * easy
    else:
        prob = 0.5
    return int(rng.random() < float(np.clip(prob, 0.02, 0.98)))


def generate_toy_frame(rows_per_group: int = 60, seed: int = 20260308) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    difficulties = ["easy", "medium", "hard"]
    for group_id, mode_id, prompt_hint in GROUPS:
        for idx in range(rows_per_group):
            difficulty = difficulties[idx % len(difficulties)]
            sample_id = f"{group_id}-{idx:04d}"
            row: dict[str, object] = {
                "sample_id": sample_id,
                "dataset_idx": f"{group_id}::{mode_id}",
                "group_id": group_id,
                "question": f"{difficulty} {prompt_hint} example {idx}",
                "answer": f"{group_id}-{difficulty}",
                "image_paths": f"{group_id}_{difficulty}_{idx}.png",
            }
            for model_id, spec in MODEL_SPECS.items():
                correct = _quality(model_id, group_id, difficulty, rng)
                row[f"{model_id}_prediction"] = f"{model_id}:{correct}"
                row[f"{model_id}_correct"] = correct
                row[f"{model_id}_token"] = int(spec["tokens"] + rng.integers(0, 24))
                row[f"{model_id}_cost"] = float(spec["cost"])
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small synthetic MMR-style routing dataset.")
    parser.add_argument("--output", default="data/examples/toy_mmr.csv", help="Output CSV path.")
    parser.add_argument("--rows-per-group", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260308)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_toy_frame(rows_per_group=args.rows_per_group, seed=args.seed)
    frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
