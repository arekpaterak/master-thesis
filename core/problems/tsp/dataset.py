from __future__ import annotations

import pickle
from pathlib import Path

import torch

from . import TSPInstance


def load_tsp_dataset(path: str | Path) -> TSPInstance:
    """Load a benchmark TSP dataset file into a single batched ``TSPInstance``.

    The file is a pickle holding a list of instances, where each instance is a
    list of ``(x, y)`` coordinate pairs in ``[0, 1]``.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)

    if not raw:
        raise ValueError(f"Empty TSP dataset: {path}")

    locations = []
    num_nodes: int | None = None
    for index, entry in enumerate(raw):
        location_tensor = torch.tensor(entry, dtype=torch.float32)

        if location_tensor.ndim != 2 or location_tensor.shape[-1] != 2:
            raise ValueError(
                f"Instance {index}: expected locations of shape (N, 2), "
                f"got {tuple(location_tensor.shape)}."
            )

        instance_num_nodes = location_tensor.shape[0]
        if num_nodes is None:
            num_nodes = instance_num_nodes
        elif instance_num_nodes != num_nodes:
            raise ValueError(
                f"Instance {index}: inconsistent number of nodes "
                f"({instance_num_nodes} vs {num_nodes}). Variable-size files are "
                "not supported."
            )

        locations.append(location_tensor)

    return TSPInstance.create(locations=torch.stack(locations, dim=0))
