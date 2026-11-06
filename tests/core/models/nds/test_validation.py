from __future__ import annotations

import pytest
import torch

from core.lns.acceptance_criterion import (
    AugmentedExchangeConfig,
    SimulatedAnnealingAcceptanceConfig,
)
from core.lns.apply_and_repair import TorchApplyAndRepair
from core.lns.config import SearchParams, StartSolutionParams
from core.lns.improvement_step import ImprovementStep
from core.models.nds.destroy import ModelNodeDestroy
from core.models.nds.models.joint import JointNDSModel
from core.models.nds.validation import EpochValidationAccumulator, NDSValidationStep
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.lns.operators.repair.reconnect import CVRPGreedyReconnectRepair
from core.problems.cvrp.nds.components import NDS as CVRP_NDS

B, N, R, DECONSTRUCTION_SIZE, STEPS = 2, 8, 4, 3, 3


@pytest.fixture(scope="module")
def model() -> JointNDSModel:
    return JointNDSModel(components=CVRP_NDS)


def _make_instance() -> CVRPInstance:
    torch.manual_seed(0)
    locations = torch.rand(B, N + 1, 2)
    demand = torch.cat([torch.zeros(B, 1), torch.rand(B, N) + 1.0], dim=-1)
    capacity = torch.ones(B, 1) * (N * 1.5)
    return CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)


def _make_validation_step(model: JointNDSModel, n_augmentations: int) -> NDSValidationStep:
    search_params = SearchParams(
        num_improvement_steps=STEPS,
        num_rollouts=R,
        deconstruction_size=DECONSTRUCTION_SIZE,
        n_augmentations=n_augmentations,
        acceptance=SimulatedAnnealingAcceptanceConfig(
            augmented_exchange=AugmentedExchangeConfig(enabled=False)
        ),
        start_solution_params=StartSolutionParams(num_improvement_steps=0),
    )
    repair = CVRPGreedyReconnectRepair(deconstruction_size=DECONSTRUCTION_SIZE)
    improvement_step = ImprovementStep(
        destroy=ModelNodeDestroy(model=model, deconstruction_size=DECONSTRUCTION_SIZE),
        apply_and_repair=TorchApplyAndRepair(repair=repair, n_repairs=1),
        mode="sequential",
        num_rollouts=R,
    )
    return NDSValidationStep(
        generate_start_solution=CVRPGraphSolution.one_customer_per_route,
        improvement_step=improvement_step,
        search_params=search_params,
    )


def test_validation_without_augmentation_reports_no_augmented_objective(model: JointNDSModel):
    summary = _make_validation_step(model, n_augmentations=1)(model, _make_instance(), epoch=0)

    assert summary.objective_final_augmented is None
    assert len(summary.objective_curve) == STEPS + 1
    assert summary.objective_initial == summary.objective_curve[0]


def test_validation_with_augmentation_reports_base_and_best(model: JointNDSModel):
    summary = _make_validation_step(model, n_augmentations=8)(model, _make_instance(), epoch=0)

    # The unsuffixed metrics track the base instance; the curve stays single-line.
    assert len(summary.objective_curve) == STEPS + 1
    assert summary.objective_initial == summary.objective_curve[0]
    # Best over 8 augmentations can never be worse than the base augmentation alone.
    assert summary.objective_final_augmented is not None
    assert summary.objective_final_augmented <= summary.objective_final


def test_validation_snapshots_follow_the_base_instance(model: JointNDSModel):
    instance = _make_instance()
    summary = _make_validation_step(model, n_augmentations=8)(model, instance, epoch=0)

    # Snapshots carry the untransformed coordinates, not a rotated augmentation's.
    assert len(summary.snapshots) == STEPS + 1
    for snapshot in summary.snapshots:
        assert torch.allclose(snapshot.instance.locations, instance.locations[0])


def test_accumulator_averages_augmented_objective_over_batches(model: JointNDSModel):
    validation_step = _make_validation_step(model, n_augmentations=8)
    accumulator = EpochValidationAccumulator()
    summaries = [validation_step(model, _make_instance(), epoch=0) for _ in range(2)]
    for summary in summaries:
        accumulator.update(summary)

    epoch_summary = accumulator.complete(epoch=0)

    expected = sum(s.objective_final_augmented for s in summaries) / len(summaries)
    assert epoch_summary.objective_final_augmented == pytest.approx(expected)
