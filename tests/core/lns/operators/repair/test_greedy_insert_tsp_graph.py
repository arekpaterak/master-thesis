import pytest
import torch

from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.insert.tsp import TSPGreedyInsertRepair
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.solve import random_permutation


@pytest.fixture(params=[3, 5])
def destruction_size(request) -> int:
    return request.param


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    return request.param


@pytest.fixture
def tsp_graph_solution(batch_size: int) -> TSPGraphSolution:
    config = TSP(num_locations=20)
    instance = TSPSampler(config, seed=42)(batch_size)
    torch.manual_seed(42)
    sequence = random_permutation(instance).tour
    return TSPGraphSolution.create_from_sequence(instance=instance, sequence=sequence)


@pytest.fixture
def deconstructed_tsp_graph_solution(
    tsp_graph_solution: TSPGraphSolution, destruction_size: int
) -> TSPGraphSolution:
    destroy = RandomNodeDestroy(destruction_size=destruction_size, seed=42, stitch=True)
    return destroy(tsp_graph_solution)


class TestTSPGreedyInsertRepair:
    def test_repairs_to_complete(self, deconstructed_tsp_graph_solution: TSPGraphSolution) -> None:
        assert not deconstructed_tsp_graph_solution.is_complete.any()

        repair = TSPGreedyInsertRepair()
        repaired = repair(deconstructed_tsp_graph_solution)

        assert repaired.is_complete.all(), "Repaired solution must be complete."

    def test_already_complete_is_unchanged(self, tsp_graph_solution: TSPGraphSolution) -> None:
        repaired = TSPGreedyInsertRepair()(tsp_graph_solution)

        assert repaired.is_complete.all()
        assert torch.allclose(repaired.objective(), tsp_graph_solution.objective())

    def test_repair_is_stochastic(self) -> None:
        """Different seeds must produce different solutions so n_repairs > 1 is meaningful.

        Uses a large destruction size relative to instance size, so insertion order
        actually interacts between free nodes and the pick order matters.
        """
        config = TSP(num_locations=20)
        instance = TSPSampler(config, seed=42)(4)
        torch.manual_seed(42)
        sequence = random_permutation(instance).tour
        solution = TSPGraphSolution.create_from_sequence(instance=instance, sequence=sequence)
        destroy = RandomNodeDestroy(destruction_size=15, seed=42, stitch=True)
        deconstructed = destroy(solution)

        results = [TSPGreedyInsertRepair(seed=seed)(deconstructed).next_nodes for seed in range(5)]
        assert any(not torch.equal(results[0], other) for other in results[1:]), (
            "Different seeds produced identical solutions; n_repairs > 1 would be wasted work."
        )

    def test_no_self_loops(self, deconstructed_tsp_graph_solution: TSPGraphSolution) -> None:
        repaired = TSPGreedyInsertRepair()(deconstructed_tsp_graph_solution)

        n = repaired.instance.num_nodes
        node_indices = torch.arange(n, device=repaired.instance.locations.device)
        assert (repaired.next_nodes != node_indices).all(), "No node should point to itself."

    def test_objective_matches_manual_traversal(
        self, deconstructed_tsp_graph_solution: TSPGraphSolution
    ) -> None:
        repaired = TSPGreedyInsertRepair()(deconstructed_tsp_graph_solution)

        B = repaired.batch_size[0]
        reference = torch.zeros(B)

        for b in range(B):
            single = repaired[b]
            locs = single.instance.locations
            n = single.instance.num_nodes
            total = 0.0
            node = 0
            for _ in range(n):
                nxt = single.next_nodes[node].item()
                total += (locs[node] - locs[nxt]).pow(2).sum().sqrt().item()
                node = nxt
            reference[b] = total

        torch.testing.assert_close(repaired.objective(), reference)

    def test_inserts_at_arbitrary_position(self) -> None:
        """Repair must be able to insert mid-tour, not only at fragment ends."""
        config = TSP(num_locations=5)
        instance = TSPSampler(config, seed=0)(1)
        # Tour 0 -> 1 -> 2 -> 3 -> 4 -> 0.
        solution = TSPGraphSolution.create_from_sequence(
            instance, torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        )
        # Detach node 2 with stitching: tour becomes 0 -> 1 -> 3 -> 4 -> 0, node 2 free.
        free = solution.remove_nodes(torch.tensor([[2]]), stitch=True)

        repaired = TSPGreedyInsertRepair(seed=0)(free)

        assert repaired.is_complete.all()
        # Node 2 must be reinserted somewhere between two distinct other nodes, not left
        # dangling or self-looped.
        assert repaired.prev_nodes[0, 2].item() != TSPGraphSolution.UNASSIGNED
        assert repaired.next_nodes[0, 2].item() != TSPGraphSolution.UNASSIGNED
        assert repaired.next_nodes[0, 2].item() != 2
        assert repaired.prev_nodes[0, 2].item() != 2
