from __future__ import annotations

from typing import Callable

from latentrouter.routers.base import BaseRouter

ROUTER_REGISTRY: dict[str, type[BaseRouter]] = {}


def register_router(name: str) -> Callable[[type[BaseRouter]], type[BaseRouter]]:
    def decorator(cls: type[BaseRouter]) -> type[BaseRouter]:
        ROUTER_REGISTRY[name] = cls
        return cls

    return decorator


def create_router(
    name: str,
    model_ids: list[str],
    random_seed: int = 20260308,
    **hyperparameters: object,
) -> BaseRouter:
    try:
        router_cls = ROUTER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown router '{name}'. Available routers: {sorted(ROUTER_REGISTRY)}") from exc
    return router_cls(model_ids=model_ids, random_seed=random_seed, **hyperparameters)
