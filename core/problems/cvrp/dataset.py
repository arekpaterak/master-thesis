from __future__ import annotations

import pickle
from pathlib import Path

import torch

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
