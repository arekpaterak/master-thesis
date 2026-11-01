import pytest
import torch

from core.problems.tsp import TSPDistribution, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.solve import random_permutation_graph


def test_create_from_sequence():
    instance = TSPSampler(TSPDistribution(num_locations=4))(3)
    sequence = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [2, 0, 3, 1],
        ]
    )

    solution = TSPGraphSolution.create_from_sequence(
        instance=instance,
        sequence=sequence,
    )

    assert torch.allclose(
        solution.next_nodes,
        torch.tensor(
            [
                [1, 2, 3, 0],
                [3, 0, 1, 2],
                [3, 2, 0, 1],
            ]
        ),
    )
    assert torch.allclose(
        solution.prev_nodes,
        torch.tensor(
            [
                [3, 0, 1, 2],
                [1, 2, 3, 0],
                [2, 3, 1, 0],
            ]
        ),
    )


def test_fail_to_create_with_unassigned_nodes():
    instance = TSPSampler(TSPDistribution(num_locations=4))(2)
    sequence = torch.tensor(
        [
            [0, 1, -1, 3],
            [3, 2, 1, 0],
        ]
    )

    with pytest.raises(Exception):
        TSPGraphSolution.create_from_sequence(instance=instance, sequence=sequence)


@pytest.mark.parametrize(
    "sequence, to_nodes, expected_next, expected_prev",
    [
        (
            [[0, 1, 2, 3]],
            [1],
            [-1, 2, 3, 0],
            [3, -1, 1, 2],
        ),
        (
            [[0, 1, 2, 3]],
            [[1]],
            [-1, 2, 3, 0],
            [3, -1, 1, 2],
        ),
        (
            [[0, 1, 2, 3]],
            [[1, 0]],
            [-1, 2, 3, -1],
            [-1, -1, 1, 2],
        ),
        (
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[1, 0], [1, 0]],
            [[-1, 2, 3, -1], [-1, 2, 3, -1]],
            [[-1, -1, 1, 2], [-1, -1, 1, 2]],
        ),
    ],
)
def test_remove_edges(sequence, to_nodes, expected_next, expected_prev):
    instance = TSPSampler(TSPDistribution(num_locations=4))(len(sequence))
    solution = TSPGraphSolution.create_from_sequence(
        instance=instance, sequence=torch.tensor(sequence)
    )

    solution.remove_edges(end_nodes=torch.tensor(to_nodes))

    assert torch.allclose(solution.next_nodes, torch.tensor(expected_next))
    assert torch.allclose(solution.prev_nodes, torch.tensor(expected_prev))


def test_add_edges():
    instance = TSPSampler(TSPDistribution(num_locations=4))(2)

    solution = TSPGraphSolution.create_from_sequence(instance)

    edges = torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[3, 0], [1, 2]],
        ]
    )

    solution.add_edges(edges)

    assert solution.next_nodes[0, 0] == 1
    assert solution.next_nodes[0, 2] == 3
    assert solution.next_nodes[1, 3] == 0
    assert solution.next_nodes[1, 1] == 2

    assert solution.prev_nodes[0, 1] == 0
    assert solution.prev_nodes[0, 3] == 2
    assert solution.prev_nodes[1, 0] == 3
    assert solution.prev_nodes[1, 2] == 1

    # Unassigned slots are untouched
    assert (solution.next_nodes[0, [1, 3]] == TSPGraphSolution.UNASSIGNED).all()
    assert (solution.prev_nodes[0, [0, 2]] == TSPGraphSolution.UNASSIGNED).all()


class TestMultiDimBatch:
    """TSPGraphSolution methods must work for batch shapes beyond 1D (e.g. (B, R) from rollouts)."""

    @pytest.fixture
    def solution_2d(self):
        B, R, N, K = 3, 5, 10, 2
        instance = TSPSampler(TSPDistribution(num_locations=N))(B)
        flat = random_permutation_graph(instance)  # (B,) batch
        # Use first K nodes per element
        nodes = torch.arange(K).unsqueeze(0).unsqueeze(0).expand(B, R, K)
        return flat.remove_nodes(nodes)  # (B, R) batch

    def test_remove_nodes_batch_shape(self, solution_2d):
        assert solution_2d.batch_size == torch.Size([3, 5])

    def test_neighbours_index_shape(self, solution_2d):
        idx = solution_2d.neighbours_index()
        assert idx.shape == torch.Size([3, 5, 10, 2])
