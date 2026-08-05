import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance, SimulatedAnnealingAcceptance
from core.lns.improvement_step import NumbaSequentialImprovementStep
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.insert.tsp import TSPGreedyInsertRepair
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.solve import random_permutation_graph


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    return request.param


@pytest.fixture
def solution(batch_size: int) -> TSPGraphSolution:
    instance = TSPSampler(TSP(num_locations=20), seed=42)(batch_size)
    return random_permutation_graph(instance)


def _make_improvement_step(
    num_rollouts: int = 4, n_repairs: int = 3
) -> NumbaSequentialImprovementStep:
    destroy = RandomNodeDestroy[TSPGraphSolution](destruction_size=10, seed=42)
    return NumbaSequentialImprovementStep(
        destroy=destroy,
        repair=TSPGreedyInsertRepair(),
        num_rollouts=num_rollouts,
        n_repairs=n_repairs,
        seed=7,
    )


class TestTSPNumbaSequentialImprovementStep:
    def test_complete(self, solution: TSPGraphSolution) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert repaired.batch_size == solution.batch_size
        assert repaired.is_complete.all()

    def test_every_node_has_degree_of_two(self, solution: TSPGraphSolution) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert (repaired.node_degree() == 2).all()

    def test_objective_is_finite_and_positive(self, solution: TSPGraphSolution) -> None:
        objective = _make_improvement_step()(solution, GreedyAcceptance()).objective()

        assert objective.isfinite().all()
        assert (objective > 0).all()

    def test_objective_is_monotonic_with_greedy_acceptance(
        self, solution: TSPGraphSolution
    ) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert (repaired.objective() <= solution.objective() + 1e-4).all()

    def test_with_simulated_annealing(self, solution: TSPGraphSolution) -> None:
        criterion = SimulatedAnnealingAcceptance(temperature=0.1, decay=0.9)
        repaired = _make_improvement_step()(solution, criterion)

        assert repaired.is_complete.all()
        assert (repaired.node_degree() == 2).all()

    def test_with_multidim_batch(self) -> None:
        # random_permutation_graph only supports a single leading batch dim, so build
        # the (identity) starting tour directly for this multi-dim-batch instance.
        instance = TSPSampler(TSP(num_locations=20), seed=1)(3).augment(2)
        sequence = torch.arange(instance.num_locations).expand(*instance.batch_size, -1)
        solution = TSPGraphSolution.create_from_sequence(instance=instance, sequence=sequence)
        assert solution.batch_size == torch.Size([3, 2])

        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert repaired.batch_size == torch.Size([3, 2])
        assert repaired.is_complete.all()

    def test_if_deterministic_for_fixed_seed(self, solution: TSPGraphSolution) -> None:
        first = _make_improvement_step()(solution, GreedyAcceptance())
        second = _make_improvement_step()(solution, GreedyAcceptance())

        assert torch.equal(first.next_nodes, second.next_nodes)

    def test_insert_in_affected_routes_only_is_unsupported(
        self, solution: TSPGraphSolution
    ) -> None:
        destroy = RandomNodeDestroy[TSPGraphSolution](destruction_size=10, seed=42)
        step = NumbaSequentialImprovementStep(
            destroy=destroy,
            repair=TSPGreedyInsertRepair(),
            num_rollouts=4,
            n_repairs=3,
            seed=7,
            insert_in_affected_routes_only=True,
        )

        with pytest.raises(NotImplementedError):
            step(solution, GreedyAcceptance())
