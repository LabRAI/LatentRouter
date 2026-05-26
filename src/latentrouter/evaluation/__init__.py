from latentrouter.evaluation.metrics import best_single_model, compute_nauc, compute_ps, compute_qnc

__all__ = ["best_single_model", "compute_nauc", "compute_ps", "compute_qnc", "evaluate_router", "run_router_on_benchmark"]


def __getattr__(name: str):
    if name == "evaluate_router":
        from latentrouter.evaluation.runner import evaluate_router

        return evaluate_router
    if name == "run_router_on_benchmark":
        from latentrouter.evaluation.runner import run_router_on_benchmark

        return run_router_on_benchmark
    raise AttributeError(name)
