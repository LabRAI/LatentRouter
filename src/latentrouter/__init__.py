"""latentrouter package."""

from latentrouter.cli import main
from latentrouter.data.normalize import prepare_benchmark, prepare_dataset
from latentrouter.evaluation.runner import run_router_on_benchmark

__all__ = ["main", "prepare_benchmark", "prepare_dataset", "run_router_on_benchmark"]
