import pytest
import torch

from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.lns.operators.repair.reconnect import TSPGreedyReconnectRepair
from core.problems.tsp.solve import random_permutation
from tests.core.lns.operators.repair.templates import RepairContract


@pytest.fixture(params=[3, 5])
def destruction_size(request) -> int:
    return request.param


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    return request.param


def _graph_solution(batch_size: int, num_locations: int = 20) -> TSPGraphSolution:
    config = TSP(num_locations=num_locations)
    instance = TSPSampler(config, seed=42)(batch_size)
    torch.manual_seed(42)
    sequence = random_permutation(instance).tour
    return TSPGraphSolution.create_from_sequence(instance=instance, sequence=sequence)


class TestTSPGreedyReconnectRepair(RepairContract):
    @pytest.fixture
    def complete_solution(self, batch_size: int) -> TSPGraphSolution:
        return _graph_solution(batch_size)

    @pytest.fixture
    def deconstructed(
        self, complete_solution: TSPGraphSolution, destruction_size: int
    ) -> TSPGraphSolution:
        destroy = RandomNodeDestroy[TSPGraphSolution](destruction_size=destruction_size, seed=42)
        return destroy(complete_solution)

    def make_repair(self, *, seed: int | None = None) -> TSPGreedyReconnectRepair:
        return TSPGreedyReconnectRepair(seed=seed)

    # -- TSP specifics ----------------------------------------------------------

    def test_repair_is_stochastic(self) -> None:
        """Repeated repair calls must produce different solutions so n_repairs > 1 is meaningful."""
        deconstructed = RandomNodeDestroy[TSPGraphSolution](destruction_size=8, seed=42)(
            _graph_solution(batch_size=4)
        )

        first = self.make_repair(seed=0)(deconstructed)
        second = self.make_repair(seed=1)(deconstructed)

        assert first.is_complete.all() and second.is_complete.all()
        assert not torch.equal(first.next_nodes, second.next_nodes), (
            "tsp reconnect repair produced identical solutions across seeds; "
            "n_repairs > 1 would be wasted work."
        )

    @pytest.mark.parametrize(
        "batch_size, num_locations, destruction_size",
        [
            (1, 10, 3),
            (4, 20, 5),
            (4, 20, 2),  # adjacent removals likely in some batch elements
        ],
    )
    def test_node_destroy_severs_expected_edge_count(
        self, batch_size: int, num_locations: int, destruction_size: int
    ) -> None:
        solution = _graph_solution(batch_size, num_locations)
        assert solution.is_complete.all()

        destroyed = RandomNodeDestroy(destruction_size=destruction_size, seed=42)(solution)

        assert not destroyed.is_complete.any()
        # Each removed node severs 2 edges, so num_unassigned is between
        # destruction_size + 1 (all adjacent) and 2 * destruction_size (all non-adjacent).
        assert (destroyed.num_unassigned >= destruction_size + 1).all()
        assert (destroyed.num_unassigned <= 2 * destruction_size).all()

        repaired = self.make_repair()(destroyed)
        assert repaired.is_complete.all()
