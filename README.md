# LatentRouter

LatentRouter is an offline multimodal model router. Given an image-question query and a pool of candidate MLLMs, it predicts a counterfactual utility for each model and routes to the model with the highest predicted utility. The implementation in this repository focuses on the proposed method, not on shipping benchmark data or third-party contender implementations.

This repository is prepared for anonymous review. It intentionally excludes paper PDFs, author metadata, local paths, raw benchmark data, generated artifacts, and private experiment logs.

## Method Overview

LatentRouter treats model selection as counterfactual multimodal utility prediction:

1. Query features are projected into learned multimodal routing capsules.
2. Candidate MLLMs are represented by capability tokens built from training-split calibration profiles, normalized cost, latency, slice behavior, and pairwise behavior.
3. Latent communication layers let model tokens read query capsules, compare against other available models with availability masking, and send feedback to the query capsules.
4. A shared distributional outcome head predicts per-model quality mean and uncertainty.
5. A bounded capsule correction refines close choices while limiting residual magnitude.
6. The router selects `argmax_i predicted_quality_i - lambda * normalized_cost_i`.

The same scoring head is shared across model tokens, so unavailable models can be masked and the router can evaluate changed candidate pools without a fixed K-way classifier.

## Repository Organization

```text
configs/                         YAML configs for toy and benchmark-style runs
scripts/generate_toy_mmr.py       Small synthetic MMR-style data generator
src/latentrouter/benchmarks/      MMR-Bench and VL-RouterBench normalization/evaluation adapters
src/latentrouter/data/            Split and normalization entrypoints
src/latentrouter/embedding/       Hashing and optional OpenCLIP feature builders
src/latentrouter/evaluation/      Protocol dispatch and metrics
src/latentrouter/memory/          Train-only model capability profile construction
src/latentrouter/routers/         LatentRouter implementation and registry
tests/                            Release smoke test
```

## Requirements

Python 3.10 or newer is recommended. The default smoke path trains the router on CUDA. A CUDA-capable PyTorch install is expected for smoke tests and benchmark reproduction.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[hf]"      # Hugging Face dataset loading
pip install -e ".[vision]"  # OpenCLIP image/text features
```

## Quick Start

Generate a small synthetic MMR-style routing dataset:

```bash
python scripts/generate_toy_mmr.py --output data/examples/toy_mmr.csv
```

Run LatentRouter end to end:

```bash
latentrouter --config configs/toy.yaml run \
  --source data/examples/toy_mmr.csv \
  --processed-dir data/processed/toy_mmr \
  --model-path artifacts/models/toy_latentrouter.pkl \
  --run-dir artifacts/runs/toy_latentrouter \
  --force \
  --force-retrain
```

The command writes normalized artifacts, cached features, a trained router checkpoint, `metrics.json`, `summary.md`, and frontier plots under `artifacts/`. The toy config sets `router.hyperparameters.device: cuda`; the hashing feature backend is lightweight, but the trainable LatentRouter itself runs on GPU.

## Running On Benchmark Data

The release does not commit benchmark datasets because they are external assets, but the code runs directly from the original public sources.

MMR-Bench source: https://huggingface.co/datasets/gh0stHunter/MMR-Bench

```bash
pip install -e ".[hf]"
latentrouter --config configs/default.yaml run \
  --benchmark mmr \
  --source hf://gh0stHunter/MMR-Bench \
  --source-split train \
  --processed-dir data/processed/mmr_bench \
  --backend hashing \
  --model-path artifacts/models/mmr_latentrouter.pkl \
  --run-dir artifacts/runs/mmr_latentrouter \
  --force \
  --force-retrain
```

VL-RouterBench sources: https://github.com/K1nght/VL-RouterBench and https://huggingface.co/datasets/KinghtH/VL-RouterBench

```bash
pip install -e ".[hf]"
mkdir -p data/raw/vl_routerbench
huggingface-cli download KinghtH/VL-RouterBench vlm_router_data.tar.gz \
  --repo-type dataset \
  --local-dir data/raw/vl_routerbench

latentrouter --config configs/default.yaml run \
  --benchmark vl_routerbench \
  --source data/raw/vl_routerbench/vlm_router_data.tar.gz \
  --processed-dir data/processed/vl_routerbench \
  --backend hashing \
  --model-path artifacts/models/vl_latentrouter.pkl \
  --run-dir artifacts/runs/vl_latentrouter \
  --force \
  --force-retrain
```

Both commands use the default CUDA router setting. For richer multimodal features, install the `vision` extra and set `--backend openclip`; keep the same feature directory for every method you compare.

The MMR adapter also accepts a local CSV/parquet file with one row per query and per-model columns such as:

```text
sample_id,dataset_idx,question,answer,image_paths,
model_a_prediction,model_a_correct,model_a_token,model_a_cost,
model_b_prediction,model_b_correct,model_b_token,model_b_cost
```

## Reproducibility Notes

- Capability tokens are built only from the training/calibration split.
- Test outcomes are used only by the evaluator after the router has produced utilities.
- The normalized artifact contract is `samples.parquet`, `outcomes.parquet`, `models.json`, `manifest.json`, and `splits.parquet`.
- All compared methods should use the same processed directory, split policy, candidate pool, feature directory, metric protocol, and cost protocol.
- The toy dataset is a functional smoke test, not a replacement for full MMR-Bench or VL-RouterBench evaluation.

## Tests

```bash
pytest -q
```

The release smoke test generates a tiny synthetic dataset and trains the LatentRouter implementation on CUDA. CPU-only hosts skip this test by design.
