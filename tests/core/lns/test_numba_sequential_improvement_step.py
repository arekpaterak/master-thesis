import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance, SimulatedAnnealingAcceptance
from core.lns.improvement_step import NumbaSequentialImprovementStep
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.insert.cvrp import CVRPGreedyInsertRepair
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.solve import one_customer_per_route_graph


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    return request.param


@pytest.fixture
def solution(batch_size: int) -> CVRPGraphSolution:
    instance = CVRPSampler(CVRP.for_n(20), seed=42)(batch_size)
    return one_customer_per_route_graph(instance)


def _make_improvement_step(
    num_rollouts: int = 4, n_repairs: int = 3
) -> NumbaSequentialImprovementStep:
    destroy = RandomNodeDestroy[CVRPGraphSolution](destruction_size=10, seed=42, stitch=True)
    return NumbaSequentialImprovementStep(
        destroy=destroy,
        repair=CVRPGreedyInsertRepair(),
        num_rollouts=num_rollouts,
        n_repairs=n_repairs,
        seed=7,
    )


class TestNumbaSequentialImprovementStep:
    def test_complete_and_feasible(self, solution: CVRPGraphSolution) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert repaired.batch_size == solution.batch_size
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_objective_is_finite_and_positive(self, solution: CVRPGraphSolution) -> None:
        objective = _make_improvement_step()(solution, GreedyAcceptance()).objective()

        assert objective.isfinite().all()
        assert (objective > 0).all()

    def test_objective_is_monotonic_with_greedy_acceptance(
        self, solution: CVRPGraphSolution
    ) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert (repaired.objective() <= solution.objective() + 1e-4).all()

    def test_depot_invariant_preserved(self, solution: CVRPGraphSolution) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert (repaired.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (repaired.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()

    def test_every_customer_has_degree_of_two(self, solution: CVRPGraphSolution) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert (repaired.node_degree() == 2).all()

    def test_with_simulated_annealing(self, solution: CVRPGraphSolution) -> None:
        criterion = SimulatedAnnealingAcceptance(temperature=0.1, decay=0.9)
        repaired = _make_improvement_step()(solution, criterion)

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_with_multidim_batch(self) -> None:
        instance = CVRPSampler(CVRP.for_n(20), seed=1)(3).augment(2)
        solution = one_customer_per_route_graph(instance)
        assert solution.batch_size == torch.Size([3, 2])

        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert repaired.batch_size == torch.Size([3, 2])
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_if_deterministic_for_fixed_seed(self, solution: CVRPGraphSolution) -> None:
        first = _make_improvement_step()(solution, GreedyAcceptance())
        second = _make_improvement_step()(solution, GreedyAcceptance())

        assert torch.equal(first.next_nodes, second.next_nodes)

    def test_rejects_non_random_node_destroy(self) -> None:
        class _Dummy:
            pass

        with pytest.raises(NotImplementedError):
            NumbaSequentialImprovementStep(destroy=_Dummy(), repair=CVRPGreedyInsertRepair())
