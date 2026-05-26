from __future__ import annotations

from pathlib import Path

from latentrouter.io import read_json

from .base import BenchmarkAdapter, BenchmarkProtocol
from .mmr import MMRAdapter, MMRProtocol
from .vl_routerbench import VLRouterBenchAdapter, VLRouterBenchProtocol

ADAPTER_REGISTRY: dict[str, type[BenchmarkAdapter]] = {
    "mmr": MMRAdapter,
    "vl_routerbench": VLRouterBenchAdapter,
}

PROTOCOL_REGISTRY: dict[str, type[BenchmarkProtocol]] = {
    "mmr_cost_quality": MMRProtocol,
    "vl_routerbench_rank_score": VLRouterBenchProtocol,
}


def create_adapter(benchmark_id: str) -> BenchmarkAdapter:
    try:
        return ADAPTER_REGISTRY[benchmark_id]()
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark adapter: {benchmark_id}") from exc


def create_protocol(protocol_id: str) -> BenchmarkProtocol:
    try:
        return PROTOCOL_REGISTRY[protocol_id]()
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark protocol: {protocol_id}") from exc


def load_benchmark_manifest(processed_dir: str | Path) -> dict:
    return read_json(Path(processed_dir) / "manifest.json")


def load_benchmark_protocol(
    processed_dir: str | Path | None = None,
    benchmark_id: str | None = None,
    protocol_id: str | None = None,
) -> BenchmarkProtocol:
    if protocol_id is None:
        if processed_dir is not None:
            protocol_id = str(load_benchmark_manifest(processed_dir)["protocol_id"])
        elif benchmark_id == "mmr":
            protocol_id = "mmr_cost_quality"
        elif benchmark_id == "vl_routerbench":
            protocol_id = "vl_routerbench_rank_score"
        else:
            raise ValueError("Either processed_dir, benchmark_id, or protocol_id is required.")
    return create_protocol(protocol_id)
