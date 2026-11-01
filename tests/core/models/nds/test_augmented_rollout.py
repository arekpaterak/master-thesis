from __future__ import annotations

import pytest
import torch

from core.lns.acceptance_criterion import SimulatedAnnealingAcceptance
from core.lns.apply_and_repair import TorchApplyAndRepair
from core.lns.improvement_step import ImprovementStep
from core.lns.search import AugmentationExchange, search
from core.models.nds.destroy import ModelNodeDestroy
from core.models.nds.models.joint import JointNDSModel
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.lns.operators.repair.reconnect import CVRPGreedyReconnectRepair

B, A, N, R, DECONSTRUCTION_SIZE = 2, 3, 8, 4, 3


def _make_solution(*, augment: int | None) -> CVRPGraphSolution:
    torch.manual_seed(0)
    locations = torch.rand(B, N + 1, 2)
    demand = torch.cat([torch.zeros(B, 1), torch.rand(B, N) + 1.0], dim=-1)
    capacity = torch.ones(B, 1) * (N * 1.5)
    instance = CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)
    if augment is not None:
        instance = instance.augment(augment)
    return CVRPGraphSolution.one_customer_per_route(instance)


@pytest.fixture(scope="module")
def model() -> JointNDSModel:
    return JointNDSModel(problem_name="cvrp")


def test_rollout_on_augmented_solution_keeps_augmentation_dim(model: JointNDSModel):
    solution = _make_solution(augment=A)

    rollout = model.rollout(
        solution,
        rollouts_per_solution=R,
        deconstruction_size=DECONSTRUCTION_SIZE,
        greedy=False,
    )

    assert rollout.batch_size == torch.Size([B, A, R])
    assert rollout.actions.shape == (B, A, R, DECONSTRUCTION_SIZE)
    assert rollout.log_p.shape == (B, A, R)


def test_rollout_plain_path_is_unchanged(model: JointNDSModel):
    solution = _make_solution(augment=None)

    rollout = model.rollout(
        solution,
        rollouts_per_solution=R,
        deconstruction_size=DECONSTRUCTION_SIZE,
        greedy=False,
    )

    assert rollout.batch_size == torch.Size([B, R])
    assert rollout.actions.shape == (B, R, DECONSTRUCTION_SIZE)


def test_augmented_sa_search_with_model_reduces_to_batch_augmentation(model: JointNDSModel):
    solution = _make_solution(augment=A)
    repair = CVRPGreedyReconnectRepair(deconstruction_size=DECONSTRUCTION_SIZE)
    destroy = ModelNodeDestroy(model=model, deconstruction_size=DECONSTRUCTION_SIZE)
    improvement_step = ImprovementStep(
        destroy=destroy,
        apply_and_repair=TorchApplyAndRepair(repair=repair, n_repairs=1),
        mode="sequential",
        num_rollouts=R,
    )
    criterion = SimulatedAnnealingAcceptance.from_range(
        temperature_start=0.1, temperature_min=0.001, n_steps=3
    )

    result = search(
        solution,
        improvement_step=improvement_step,
        criterion=criterion,
        n_iterations=3,
        exchange=AugmentationExchange(criterion, threshold_factor=15.0),
    )

    assert result.solution.batch_size == torch.Size([B, A])
    objective = result.solution.objective()
    assert objective.shape == (B, A)
    assert torch.isfinite(objective).all()
