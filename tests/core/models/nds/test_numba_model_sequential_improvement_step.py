import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance, SimulatedAnnealingAcceptance
from core.lns.improvement_step import ImprovementStep
from core.models.nds.destroy import ModelNodeDestroy
from core.models.nds.models.joint import JointNDSModel
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.lns.apply_repair import CVRPGreedyInsertApplyRepair
from core.problems.cvrp.solve import one_customer_per_route_graph


@pytest.fixture
def model() -> JointNDSModel:
    torch.manual_seed(0)
    return JointNDSModel(problem_name="cvrp")


@pytest.fixture(params=[1, 4])
def solution(request) -> CVRPGraphSolution:
    instance = CVRPSampler(CVRPDistribution.for_n(20), seed=42)(request.param)
    return one_customer_per_route_graph(instance)


def _make_improvement_step(
    model: JointNDSModel, insert_in_affected_routes_only: bool = False
) -> ImprovementStep:
    destroy = ModelNodeDestroy(model=model, deconstruction_size=10)
    apply_and_repair = CVRPGreedyInsertApplyRepair(
        n_repairs=2, seed=7, insert_in_affected_routes_only=insert_in_affected_routes_only
    )
    return ImprovementStep(
        destroy=destroy,
        apply_and_repair=apply_and_repair,
        mode="sequential",
        num_rollouts=4,
    )


class TestNumbaModelSequentialImprovementStep:
    def test_if_complete_and_feasible(
        self, model: JointNDSModel, solution: CVRPGraphSolution
    ) -> None:
        improved = _make_improvement_step(model)(solution, GreedyAcceptance())

        assert improved.batch_size == solution.batch_size
        assert improved.is_complete.all()
        assert improved.is_feasible.all()

    def test_objective_is_monotonic_with_greedy_acceptance(
        self, model: JointNDSModel, solution: CVRPGraphSolution
    ) -> None:
        improved = _make_improvement_step(model)(solution, GreedyAcceptance())

        assert (improved.objective() <= solution.objective() + 1e-4).all()

    def test_depot_invariant_and_degrees(
        self, model: JointNDSModel, solution: CVRPGraphSolution
    ) -> None:
        improved = _make_improvement_step(model)(solution, GreedyAcceptance())

        assert (improved.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (improved.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (improved.node_degree() == 2).all()

    def test_with_simulated_annealing(
        self, model: JointNDSModel, solution: CVRPGraphSolution
    ) -> None:
        criterion = SimulatedAnnealingAcceptance(temperature=0.1, decay=0.9)
        improved = _make_improvement_step(model)(solution, criterion)

        assert improved.is_complete.all()
        assert improved.is_feasible.all()

    def test_insert_in_affected_routes_only_stays_complete_and_feasible(
        self, model: JointNDSModel, solution: CVRPGraphSolution
    ) -> None:
        step = _make_improvement_step(model, insert_in_affected_routes_only=True)

        improved = step(solution, GreedyAcceptance())

        assert improved.is_complete.all()
        assert improved.is_feasible.all()
        assert (improved.node_degree() == 2).all()
