import pytest
import torch

from core.lns.destroy import RandomSegmentDestroy
from core.problems.tsp import TSPBatchInstance
from core.problems.tsp.solution import TSPBatchSolution


@pytest.fixture
def sample_solution():
    batch_size = 3
    n = 10
    locations = torch.randn(batch_size, n, 2)

    # Create a simple sequential tour: [0, 1, 2, ..., 9]
    tour = torch.stack([torch.arange(n) for _ in range(batch_size)])
    return TSPBatchSolution(
        instance=TSPBatchInstance(locations=locations),
        tour=tour,
    )


def test_random_segment_destroy_logic(sample_solution):
    segment_size = 3
    destroy = RandomSegmentDestroy(segment_size=segment_size)

    destroyed = destroy(sample_solution)

    assert destroyed.batch_size == sample_solution.batch_size
    assert destroyed.n == sample_solution.n


@pytest.fixture
def circular_solution():
    # Batch of 1 for easy verification, 5 nodes
    locations = torch.randn(1, 5, 2)
    tour = torch.tensor([[0, 1, 2, 3, 4]])  # Sequential tour
    return TSPBatchSolution(
        instance=TSPBatchInstance(locations=locations),
        tour=tour,
    )


def test_random_segment_destroy_wrap_around(circular_solution):
    # Arrange
    segment_size = 2
    destroy = RandomSegmentDestroy(segment_size=segment_size, seed=42)

    # Act
    # We force the start to the last index (4) and remove 2 nodes
    # Expected tour: [-1, 1, 2, 3, -1] (wrapped 4 and 0)
    while True:
        destroyed = destroy(circular_solution)
        if (destroyed.tour == -1).sum().item() == segment_size:
            break  # We got the desired random segment

    # Assert
    assert destroyed.tour.shape == (1, 5)
    assert (destroyed.tour == -1).sum() == segment_size


def test_preserves_existing_holes(circular_solution):
    # Create a solution that already has a hole
    tour_with_hole = torch.tensor([[0, -1, 2, 3, 4]])
    solution = TSPBatchSolution(instance=circular_solution.instance, tour=tour_with_hole)

    destroyer = RandomSegmentDestroy(segment_size=1)
    destroyed = destroyer(solution)

    assert (destroyed.tour == -1).sum() == 2
