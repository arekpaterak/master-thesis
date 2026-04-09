import pytest
import torch

from core.problems.tsp import TSPConfig, TSPInstance, TSPSampler
from core.problems.tsp.solution import TSPSolution
from core.problems.tsp.solve import random_permutation


def test_visualize_without_batch_dim():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)[0]
    solution = random_permutation(instance)
    assert solution.batch_dims == 0

    solution.visualize()


def test_visualize_with_batch_of_one():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)[:1]
    solution = random_permutation(instance)
    assert solution.batch_dims == 1

    solution.visualize()


def test_if_fails_to_visualize_batch():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)
    solution = random_permutation(instance)

    with pytest.raises(ValueError):
        solution.visualize()


@pytest.mark.parametrize(
    "locations,tour,expected",
    [
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [0, 1, 2, 3], [4.0]),
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [0, 1, 2], ValueError),
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [0, 1, -1, 2], ValueError),
    ],
)
def test_objective(locations, tour, expected):
    locations = torch.tensor(locations, dtype=torch.float)
    tour = torch.tensor(tour, dtype=torch.long)
    instance = TSPInstance.create(
        locations=locations,
    )
    solution = TSPSolution.create(
        instance=instance,
        tour=tour,
    )

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            solution.objective()
        return

    objective = solution.objective()

    expected = torch.tensor(expected, dtype=torch.float)
    assert (objective > 0).all()
    assert torch.allclose(objective, expected)


@pytest.mark.parametrize(
    "locations,tour,expected",
    [
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [0, 1, 2, 3], [4.0]),
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [0, 1, 2], [2.0]),
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [0, 1, -1, 2], [1.0]),
    ],
)
def test_tour_length(locations, tour, expected):
    locations = torch.tensor(locations, dtype=torch.float)
    tour = torch.tensor(tour, dtype=torch.long)
    instance = TSPInstance.create(
        locations=locations,
    )
    solution = TSPSolution.create(
        instance=instance,
        tour=tour,
    )

    tour_length = solution.tour_length()

    expected = torch.tensor(expected, dtype=torch.float)
    assert (tour_length > 0).all()
    assert torch.allclose(tour_length, expected)


def test_deconstructed():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)
    solution = random_permutation(instance)

    solution[0, 1] = -1

    assert (solution.is_complete == False)[0]
    assert (solution.is_complete == True)[1:].all()

    solution[:, 1] = -1

    assert (solution.is_complete == False).all()
