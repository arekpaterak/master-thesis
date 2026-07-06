from __future__ import annotations

import torch

from core.lns.acceptance_criterion import SimulatedAnnealingAcceptance
from core.lns.improvement_step import NoopImprovementStep
from core.lns.search import (
    _exchange_above_threshold,
    build_augmentation_synchronizer,
    search,
)
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance


def _make_instance(B: int, N: int, A: int | None = None) -> CVRPInstance:
    torch.manual_seed(0)
    locations = torch.rand(B, N + 1, 2)
    demand = torch.cat([torch.zeros(B, 1), torch.rand(B, N) + 1.0], dim=-1)
    capacity = torch.ones(B, 1) * (N * 1.5)
    instance = CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)
    if A is not None:
        return instance.augment(A)
    return instance


def test_exchange_above_threshold_replaces_only_bad_with_good():
    B, A, N = 3, 4, 8
    instance = _make_instance(B, N, A)
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    # Synthetic costs and threshold: per-row [0.0, 1.0, 5.0, 10.0], threshold 2.0.
    costs = torch.tensor([[0.0, 1.0, 5.0, 10.0]] * B)
    threshold = torch.tensor([[2.0]] * B)

    torch.manual_seed(123)
    new_solution = _exchange_above_threshold(solution, costs, threshold)

    # All slots remain valid CVRP graph solutions (route IDs stay >= 0 for customers).
    assert (new_solution.route_ids[..., 1:] >= 0).all()
    # The two "good" slots (0 and 1) are unchanged.
    assert torch.equal(new_solution._next[:, :2], solution._next[:, :2])
    assert torch.equal(new_solution._prev[:, :2], solution._prev[:, :2])
    # The two "bad" slots (2 and 3) now point at one of the good slots' state.
    for b in range(B):
        for bad in (2, 3):
            assert torch.equal(new_solution._next[b, bad], solution._next[b, 0]) or torch.equal(
                new_solution._next[b, bad], solution._next[b, 1]
            )


def test_exchange_with_no_good_solutions_keeps_existing_solutions():
    B, A, N = 2, 3, 6
    instance = _make_instance(B, N, A)
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    costs = torch.tensor([[5.0, 5.0, 5.0]] * B)
    threshold = torch.tensor([[1.0]] * B)

    new_solution = _exchange_above_threshold(solution, costs, threshold)
    assert torch.equal(new_solution._next, solution._next)
    assert torch.equal(new_solution._prev, solution._prev)
    assert torch.equal(new_solution._route_id, solution._route_id)


def test_search_with_sa_synchronizer_preserves_augmentation_dim():
    B, A, N = 2, 4, 6
    instance = _make_instance(B, N, A)
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    criterion = SimulatedAnnealingAcceptance.from_range(
        temperature_start=0.1, temperature_min=0.001, n_steps=5
    )
    synchronize = build_augmentation_synchronizer(criterion, threshold_factor=15.0)

    result = search(
        solution,
        improvement_step=NoopImprovementStep(),
        criterion=criterion,
        n_iterations=5,
        synchronize=synchronize,
    )

    assert result.solution.batch_size == torch.Size([B, A])
    objective = result.solution.objective()
    assert objective.shape == (B, A)
    assert torch.isfinite(objective).all()


def test_sa_synchronizer_is_noop_when_all_augmented_copies_share_cost():
    # D4 augmentations are isometries, so every augmented copy of the same solution
    # has identical cost; no copy exceeds the threshold, so the exchange is a no-op
    # and the run matches plain search.
    B, A, N = 2, 4, 6
    instance = _make_instance(B, N, A)
    solution = CVRPGraphSolution.one_customer_per_route(instance)

    criterion_plain = SimulatedAnnealingAcceptance.from_range(
        temperature_start=0.1, temperature_min=0.001, n_steps=5
    )
    criterion_synced = SimulatedAnnealingAcceptance.from_range(
        temperature_start=0.1, temperature_min=0.001, n_steps=5
    )
    synchronize = build_augmentation_synchronizer(criterion_synced, threshold_factor=15.0)

    plain = search(
        solution,
        improvement_step=NoopImprovementStep(),
        criterion=criterion_plain,
        n_iterations=5,
    )
    synced = search(
        solution,
        improvement_step=NoopImprovementStep(),
        criterion=criterion_synced,
        n_iterations=5,
        synchronize=synchronize,
    )

    assert torch.equal(plain.solution._next, synced.solution._next)
    assert torch.equal(plain.solution._prev, synced.solution._prev)
