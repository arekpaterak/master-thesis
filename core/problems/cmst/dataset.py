from __future__ import annotations

from pathlib import Path

from core.problems.cvrp.dataset import load_cvrp_dataset

from . import CMSTInstance


def load_cmst_dataset(path: str | Path) -> CMSTInstance:
    """Load a benchmark CVRP dataset file as a single batched ``CMSTInstance``.

    CMST and CVRP share the same instance data (root/depot + terminals/customers +
    demand + capacity); only the solution built on top differs.
    """
    instance = load_cvrp_dataset(path)
    return CMSTInstance.create(
        locations=instance.locations,
        demand=instance.demand,
        capacity=instance.capacity,
        coordinate_scale=instance.coordinate_scale,
    )
