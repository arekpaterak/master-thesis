from __future__ import annotations

from typing import Protocol

import torch

from core.problems.cvrp import CVRPBatchInstance, CVRPBatchSolution


class CVRPSolveStrategy(Protocol):
    def __call__(self, instance: CVRPBatchInstance) -> CVRPBatchSolution: ...


def one_customer_per_route(instance: CVRPBatchInstance) -> CVRPBatchSolution:
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

    sequences = torch.stack([depots, customers], dim=2).view(B, -1)
    final_zeros = torch.zeros((B, 1), dtype=torch.long, device=device)
    sequences = torch.cat([sequences, final_zeros], dim=1)

    if not was_batched:
        instance = instance.squeeze(0)
        sequences = sequences.squeeze(0)

    return CVRPBatchSolution(instance=instance, sequences=sequences, batch_size=instance.batch_size)


def all_customers_one_route(instance: CVRPBatchInstance) -> CVRPBatchSolution:
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

    sequences = torch.cat([start, customers, end], dim=1)

    if not was_batched:
        instance = instance.squeeze(0)
        sequences = sequences.squeeze(0)

    return CVRPBatchSolution(instance=instance, sequences=sequences, batch_size=instance.batch_size)


def index_sequential(instance: CVRPBatchInstance) -> CVRPBatchSolution:
    """Add customers in index order until capacity is exhausted."""
    was_batched = True
    if len(instance.batch_size) == 0:
        instance = instance.unsqueeze(0)
        was_batched = False

    B = instance.batch_size[0]
    N = instance.num_customers
    device = instance.device

    sequences = []
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
        sequences.append(torch.tensor(tokens, dtype=torch.long, device=device))
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    if not was_batched:
        instance = instance.squeeze(0)
        sequences = sequences.squeeze(0)

    return CVRPBatchSolution(instance=instance, sequences=sequences, batch_size=instance.batch_size)


STRATEGIES = [
    one_customer_per_route,
    all_customers_one_route,
    index_sequential,
]
