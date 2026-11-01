from __future__ import annotations

from typing import Protocol

import torch

from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.solution import CVRPSolution

__all__ = [
    "STRATEGIES",
    "CVRPSolveStrategy",
    "all_customers_one_route",
    "index_sequential",
    "one_customer_per_route",
    "one_customer_per_route_graph",
]


class CVRPSolveStrategy(Protocol):
    def __call__(self, instance: CVRPInstance) -> CVRPSolution: ...


def one_customer_per_route(instance: CVRPInstance) -> CVRPSolution:
    """Each customer gets its own route. Always feasible if no single demand exceeds capacity."""
    was_batched = True
    if len(instance.batch_size) == 0:
        instance = instance.unsqueeze(0)
        was_batched = False

    B = instance.batch_size[0]
    N = instance.num_customers
    device = instance.device

    customers = torch.arange(1, N + 1, device=device).repeat(B, 1)
    depots = torch.zeros_like(customers)

    sequence = torch.stack([depots, customers], dim=2).view(B, -1)
    final_zeros = torch.zeros((B, 1), dtype=torch.long, device=device)
    sequence = torch.cat([sequence, final_zeros], dim=1)

    if not was_batched:
        instance = instance.squeeze(0)
        sequence = sequence.squeeze(0)

    return CVRPSolution.create(
        instance=instance,
        sequence=sequence,
    )


def all_customers_one_route(instance: CVRPInstance) -> CVRPSolution:
    """
    Single route visiting all customers (e.g. [0, 1, 2, ..., 0]).
    Ignores capacity so may be infeasible.
    """
    was_batched = True
    if len(instance.batch_size) == 0:
        instance = instance.unsqueeze(0)
        was_batched = False

    B = instance.batch_size[0]
    N = instance.num_customers
    device = instance.device

    customers = torch.arange(1, N + 1, device=device).expand(B, N)

    start = torch.zeros((B, 1), dtype=torch.long, device=device)
    end = torch.zeros((B, 1), dtype=torch.long, device=device)

    sequence = torch.cat([start, customers, end], dim=1)

    if not was_batched:
        instance = instance.squeeze(0)
        sequence = sequence.squeeze(0)

    return CVRPSolution.create(
        instance=instance,
        sequence=sequence,
    )


def index_sequential(instance: CVRPInstance) -> CVRPSolution:
    """Add customers in index order until capacity is exhausted."""
    was_batched = True
    if len(instance.batch_size) == 0:
        instance = instance.unsqueeze(0)
        was_batched = False

    B = instance.batch_size[0]
    N = instance.num_customers
    device = instance.device

    sequence = []
    for b in range(B):
        tokens = [0]
        load = 0.0

        for i in range(1, N + 1):
            demand = instance.demand[b, i].item()

            if load + demand > instance.capacity[b].item():
                tokens.append(0)
                load = 0.0

            tokens.append(i)
            load += demand

        tokens.append(0)
        sequence.append(torch.tensor(tokens, dtype=torch.long, device=device))
    sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=True)

    if not was_batched:
        instance = instance.squeeze(0)
        sequence = sequence.squeeze(0)

    return CVRPSolution.create(
        instance=instance,
        sequence=sequence,
    )


def one_customer_per_route_graph(instance: CVRPInstance) -> CVRPGraphSolution:
    return CVRPGraphSolution.one_customer_per_route(instance=instance)


STRATEGIES = [
    one_customer_per_route,
    all_customers_one_route,
    index_sequential,
]
