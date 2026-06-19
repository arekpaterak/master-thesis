from __future__ import annotations

import pickle
from pathlib import Path

import torch
import vrplib

from . import CVRPInstance


def load_cvrp_dataset(path: str | Path) -> CVRPInstance:
    """Load a benchmark CVRP dataset file into a single batched ``CVRPInstance``.

    The file is a pickle holding a list of ``(depot, locations, demand, capacity)``
    tuples, where demand and capacity are unnormalised. Demands are normalised by
    the capacity (depot demand set to 0) and the effective capacity becomes 1.0,
    matching the convention produced by ``CVRPSampler``.

    The returned instance lives on the CPU; callers move per-batch slices to the
    target device.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)

    if not raw:
        raise ValueError(f"Empty CVRP dataset: {path}")

    depots = []
    locations = []
    demands = []
    capacities = []
    num_nodes: int | None = None
    for index, entry in enumerate(raw):
        depot, customer_locations, demand, capacity = entry

        depot_tensor = torch.tensor(depot, dtype=torch.float32).reshape(1, 2)
        customer_tensor = torch.tensor(customer_locations, dtype=torch.float32)
        demand_tensor = torch.tensor(demand, dtype=torch.float32)
        capacity_value = float(capacity)

        if customer_tensor.ndim != 2 or customer_tensor.shape[-1] != 2:
            raise ValueError(
                f"Instance {index}: expected locations of shape (N, 2), "
                f"got {tuple(customer_tensor.shape)}."
            )
        if demand_tensor.shape[0] != customer_tensor.shape[0]:
            raise ValueError(
                f"Instance {index}: demand length {demand_tensor.shape[0]} does not "
                f"match number of customers {customer_tensor.shape[0]}."
            )

        instance_num_nodes = customer_tensor.shape[0] + 1
        if num_nodes is None:
            num_nodes = instance_num_nodes
        elif instance_num_nodes != num_nodes:
            raise ValueError(
                f"Instance {index}: inconsistent number of nodes "
                f"({instance_num_nodes} vs {num_nodes}). Variable-size files are "
                "not supported."
            )

        depot_demand = torch.zeros(1)
        normalised_demand = torch.cat([depot_demand, demand_tensor / capacity_value])

        depots.append(depot_tensor)
        locations.append(torch.cat([depot_tensor, customer_tensor], dim=0))
        demands.append(normalised_demand)
        capacities.append(torch.ones(1))

    return CVRPInstance.create(
        locations=torch.stack(locations, dim=0),
        demand=torch.stack(demands, dim=0),
        capacity=torch.stack(capacities, dim=0),
    )


def load_cvrplib_instance(path: str | Path) -> CVRPInstance:
    """Load a single CVRPLIB instance from a .vrp file into an unbatched CVRPInstance.

    Coordinates are normalized in-place via normalize_(), with the original scale
    recorded in coordinate_scale. Demands are normalized by the vehicle capacity
    (effective capacity becomes 1.0), matching the CVRPSampler convention.

    Assumes CVRP format (EUC_2D edge weights, single depot at index 0).
    """
    raw = vrplib.read_instance(str(path))
    coords = torch.tensor(raw["node_coord"], dtype=torch.float32)  # (N+1, 2)
    demand = torch.tensor(raw["demand"], dtype=torch.float32)  # (N+1,)
    capacity = float(raw["capacity"])
    instance = CVRPInstance.create(
        locations=coords,
        demand=demand / capacity,
        capacity=torch.ones(1),
    )
    instance.normalize_()
    return instance


def load_cvrplib_dataset(directory: str | Path) -> list[CVRPInstance]:
    """Load all .vrp files in a directory, returning one CVRPInstance per file.

    Files are sorted by name for reproducibility. Each instance is unbatched
    and independently normalized. Raises if the directory contains no .vrp files.
    """
    directory = Path(directory)
    paths = sorted(directory.glob("*.vrp"))
    if not paths:
        raise ValueError(f"No .vrp files found in {directory}.")
    return [load_cvrplib_instance(p) for p in paths]


def load_cvrplib_solution_bks(path: str | Path) -> int:
    """Return the BKS cost from a CVRPLIB solution file (.sol).

    The cost is the sum of rounded Euclidean distances on the original integer
    coordinates, matching CVRPSolution.total_distance(rounded=True).
    """
    return vrplib.read_solution(str(path))["cost"]
