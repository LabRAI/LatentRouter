from __future__ import annotations

from pathlib import Path

import yaml

from latentrouter.cli import main
from scripts.generate_toy_mmr import generate_toy_frame


def test_latentrouter_toy_pipeline(tmp_path: Path):
    source = tmp_path / "toy_mmr.csv"
    generate_toy_frame(rows_per_group=12, seed=7).to_csv(source, index=False)

    config_path = tmp_path / "config.yaml"
    config = {
        "data": {
            "benchmark": "mmr",
            "source": str(source),
            "processed_dir": str(tmp_path / "processed"),
            "seed": 7,
        },
        "features": {
            "backend": "hashing",
            "hashing_dim": 64,
            "force": True,
        },
        "router": {
            "name": "paper_section_3_2_latent_communication",
            "random_seed": 7,
            "hyperparameters": {
                "hidden_dim": 16,
                "num_query_tokens": 2,
                "num_capsules": 3,
                "num_layers": 1,
                "num_heads": 4,
                "batch_size": 16,
                "epochs": 2,
                "patience": 1,
                "max_calibration_samples": 24,
                "device": "cpu",
            },
        },
        "evaluation": {
            "num_lambdas": 11,
            "output_dir": str(tmp_path / "runs"),
            "aggregate_by": ["dataset_name", "mode_id"],
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = tmp_path / "run"
    model_path = tmp_path / "model.pkl"
    exit_code = main(
        [
            "--config",
            str(config_path),
            "run",
            "--source",
            str(source),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--model-path",
            str(model_path),
            "--run-dir",
            str(run_dir),
            "--force",
            "--force-retrain",
        ]
    )

    assert exit_code == 0
    assert model_path.exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "summary.md").exists()
