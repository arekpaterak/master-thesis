import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance, SimulatedAnnealingAcceptance
from core.lns.operators.repair.insert.cvrp import CVRPGreedyInsertRepair
from core.models.nds.improvement_step import (
    NumbaModelSequentialImprovementStep,
    resolve_improvement_step,
)
from core.models.nds.models.ar import AutoregressiveNDSModel
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.solve import one_customer_per_route_graph


@pytest.fixture
def model() -> AutoregressiveNDSModel:
    torch.manual_seed(0)
    return AutoregressiveNDSModel(problem_name="cvrp")


@pytest.fixture(params=[1, 4])
def solution(request) -> CVRPGraphSolution:
    instance = CVRPSampler(CVRP.for_n(20), seed=42)(request.param)
    return one_customer_per_route_graph(instance)


def _make_improvement_step(model: AutoregressiveNDSModel) -> NumbaModelSequentialImprovementStep:
    return NumbaModelSequentialImprovementStep(
        model=model,
        repair=CVRPGreedyInsertRepair(),
        num_rollouts=4,
        deconstruction_size=10,
        n_repairs=2,
        seed=7,
    )


class TestNumbaModelSequentialImprovementStep:
    def test_resolves_via_config_literal(self) -> None:
        assert resolve_improvement_step("numba_sequential") is NumbaModelSequentialImprovementStep

    def test_if_complete_and_feasible(
        self, model: AutoregressiveNDSModel, solution: CVRPGraphSolution
    ) -> None:
        improved = _make_improvement_step(model)(solution, GreedyAcceptance())

        assert improved.batch_size == solution.batch_size
        assert improved.is_complete.all()
        assert improved.is_feasible.all()

    def test_objective_is_monotonic_with_greedy_acceptance(
        self, model: AutoregressiveNDSModel, solution: CVRPGraphSolution
    ) -> None:
        improved = _make_improvement_step(model)(solution, GreedyAcceptance())

        assert (improved.objective() <= solution.objective() + 1e-4).all()

    def test_depot_invariant_and_degrees(
        self, model: AutoregressiveNDSModel, solution: CVRPGraphSolution
    ) -> None:
        improved = _make_improvement_step(model)(solution, GreedyAcceptance())

        assert (improved.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (improved.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (improved.node_degree() == 2).all()

    def test_with_simulated_annealing(
        self, model: AutoregressiveNDSModel, solution: CVRPGraphSolution
    ) -> None:
        criterion = SimulatedAnnealingAcceptance(temperature=0.1, decay=0.9)
        improved = _make_improvement_step(model)(solution, criterion)

        assert improved.is_complete.all()
        assert improved.is_feasible.all()
