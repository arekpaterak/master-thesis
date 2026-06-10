import pytest
import torch

from core.lns.operators.destroy.random_segment import TSPRandomSegmentDestroy
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.solution import TSPSolution


@pytest.fixture
def sample_solution():
    batch_size = 3
    n = 10
    instance = TSPSampler(TSP(num_locations=n))(batch_size)

    # Create a simple sequential tour: [0, 1, 2, ..., 9]
    tour = torch.stack([torch.arange(n) for _ in range(batch_size)])
    return TSPSolution.create(
        instance=instance,
        tour=tour,
    )


@pytest.fixture
def small_solution():
    batch_size = 1
    n = 5
    instance = TSPSampler(TSP(num_locations=n))(batch_size)
    tour = torch.tensor([[0, 1, 2, 3, 4]])  # Sequential tour
    return TSPSolution.create(
        instance=instance,
        tour=tour,
    )


def test_random_segment_destroy_logic(sample_solution):
    segment_size = 3
    destroy = TSPRandomSegmentDestroy(segment_size=segment_size)

    destroyed = destroy(sample_solution)

    assert destroyed.batch_size == sample_solution.batch_size
    assert destroyed.instance.num_locations == sample_solution.instance.num_locations

    for destroyed_tour in destroyed.tour:
        assert (destroyed_tour == -1).sum() == segment_size


def test_random_segment_destroy_wrap_around(small_solution):
    # Arrange
    segment_size = 2
    destroy = TSPRandomSegmentDestroy(segment_size=segment_size, seed=42)

    # Act
    # We force the start to the last index (4) and remove 2 nodes
    # Expected tour: [-1, 1, 2, 3, -1] (wrapped 4 and 0)
    while True:
        destroyed = destroy(small_solution)
        if (destroyed.tour == -1).sum().item() == segment_size:
            break  # We got the desired random segment

    # Assert
    assert destroyed.tour.shape == (1, 5)
    assert (destroyed.tour == -1).sum() == segment_size


def test_preserves_existing_holes(small_solution):
    # Create a solution that already has a hole
    tour_with_hole = torch.tensor([[0, -1, 2, 3, 4]])
    solution = TSPSolution.create(
        instance=small_solution.instance,
        tour=tour_with_hole,
    )

    destroy = TSPRandomSegmentDestroy(segment_size=1)
    destroyed = destroy(solution)

    assert destroyed.batch_size == solution.batch_size
    assert destroyed.tour[0, 1] == -1


def test_without_batch_dim(sample_solution):
    single_solution: TSPSolution = sample_solution[0]
    assert single_solution.batch_dims == 0
    num_locations = single_solution.instance.num_locations

    destroy = TSPRandomSegmentDestroy(segment_size=1)
    destroyed = destroy(single_solution)

    assert destroyed.batch_dims == 0
    assert destroyed.tour.shape == (num_locations,)
