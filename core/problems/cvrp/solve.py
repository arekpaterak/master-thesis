from __future__ import annotations

from typing import Protocol

import torch

from core.problems.cvrp import CVRPInstance, CVRPSolution


class CVRPSolveStrategy(Protocol):
    def __call__(self, instance: CVRPInstance) -> CVRPSolution: ...


def one_customer_per_route(instance: CVRPInstance) -> CVRPSolution:
    """Each customer gets its own route. Always feasible if no single demand exceeds capacity."""
    # [0, 1, 0, 2, 0, 3, 0, ...]
    customers = torch.arange(1, instance.num_customers + 1)
    depots = torch.zeros(instance.num_customers, dtype=torch.long)
    sequence = torch.stack([depots, customers], dim=1).flatten()
    sequence = torch.cat([sequence, torch.tensor([0])])
    return CVRPSolution(instance=instance, sequence=sequence)


def all_customers_one_route(instance: CVRPInstance) -> CVRPSolution:
    """Single route visiting all customers. Ignores capacity so may be infeasible."""
    customers = torch.arange(1, instance.num_customers + 1)
    sequence = torch.cat([torch.tensor([0]), customers, torch.tensor([0])])
    return CVRPSolution(instance=instance, sequence=sequence)


def index_sequential(instance: CVRPInstance) -> CVRPSolution:
    """Add customers in index order until capacity is exhausted."""
    tokens = [0]
    load = 0.0
    for i in range(1, instance.num_customers + 1):
        d = instance.demands[i].item()
        if load + d > instance.capacity:
            tokens.append(0)
            load = 0.0
        tokens.append(i)
        load += d
    tokens.append(0)
    return CVRPSolution(instance=instance, sequence=torch.tensor(tokens))
