from __future__ import annotations

import torch

from core.lns.acceptance_criterion import (
    GreedyAcceptanceConfig,
    SimulatedAnnealingAcceptanceConfig,
)
from core.lns.config import SearchParams
from core.lns.improvement_step import NoopImprovementStep
from core.methods.base import Method
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.solve import solve

B, N = 3, 6


def _make_instance() -> CVRPInstance:
    torch.manual_seed(0)
    locations = torch.rand(B, N + 1, 2)
    demand = torch.cat([torch.zeros(B, 1), torch.rand(B, N) + 1.0], dim=-1)
    capacity = torch.ones(B, 1) * (N * 1.5)
    return CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)


def _make_method(search: SearchParams) -> Method:
    return Method(
        search=search,
        generate_start_solution=CVRPGraphSolution.one_customer_per_route,
        improvement_step=NoopImprovementStep(),
    )


def test_greedy_with_augmentation_reduces_to_batch():
    method = _make_method(
        SearchParams(
            num_improvement_steps=5,
            n_augmentations=8,
            acceptance=GreedyAcceptanceConfig(),
        )
    )

    solution = solve(method, _make_instance())

    assert solution.batch_size == torch.Size([B])
    assert torch.isfinite(solution.objective()).all()


def test_simulated_annealing_with_augmentation_reduces_to_batch():
    method = _make_method(
        SearchParams(
            num_improvement_steps=5,
            n_augmentations=8,
            acceptance=SimulatedAnnealingAcceptanceConfig(),
        )
    )

    solution = solve(method, _make_instance())

    assert solution.batch_size == torch.Size([B])
    assert torch.isfinite(solution.objective()).all()


def test_single_augmentation_keeps_plain_batch():
    method = _make_method(
        SearchParams(
            num_improvement_steps=5,
            n_augmentations=1,
            acceptance=GreedyAcceptanceConfig(),
        )
    )

    solution = solve(method, _make_instance())

    assert solution.batch_size == torch.Size([B])
    assert torch.isfinite(solution.objective()).all()
