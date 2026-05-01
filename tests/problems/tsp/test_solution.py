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

    tour_length = solution._tour_length()

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


@pytest.mark.parametrize(
    "tour,to_remove,expected",
    [
        ([[0, 1, 2, 3]], [[[1, 3], [2, 1]]], [[[0, -1, 2, -1], [0, -1, -1, 3]]]),
        ([[0, 1, 2, 3]], [[1]], [[0, -1, 2, 3]]),
        ([[0, -1, 2, 3]], [[1, 2]], [[0, -1, -1, 3]]),
        ([[0, 1, 2, 3]], [[0, 4]], [[-1, 1, 2, 3]]),
    ],
)
def test_remove_nodes(tour, to_remove, expected):
    tour = torch.tensor(tour, dtype=torch.long)
    to_remove = torch.tensor(to_remove, dtype=torch.long)
    expected = torch.tensor(expected, dtype=torch.long)
    config = TSPConfig(num_locations=10)
    instance = TSPSampler(config, seed=0)(tour.shape[0])
    solution = TSPSolution.create(instance=instance, tour=tour)

    deconstructed_solution = solution.remove_nodes(to_remove)

    assert torch.equal(deconstructed_solution.tour, expected)
    assert not deconstructed_solution.is_complete.any()
    assert deconstructed_solution.instance.batch_size == deconstructed_solution.batch_size
    assert deconstructed_solution.instance.locations.shape[:-2] == deconstructed_solution.batch_size


@pytest.mark.parametrize(
    "tour,expected",
    [
        ([[0, 1, 2, 3]], [[[3, 1], [0, 2], [1, 3], [2, 0]]]),
        ([[0, 1, -1, 3]], [[[3, 1], [0, -1], [-1, -1], [-1, 0]]]),
        ([[-1, -1, -1, -1]], [[[-1, -1], [-1, -1], [-1, -1], [-1, -1]]]),
        ([[3, 2, 1, 0]], [[[1, 3], [2, 0], [3, 1], [0, 2]]]),
    ],
)
def test_neighbours_index(tour, expected):
    tour = torch.tensor(tour, dtype=torch.long)
    expected = torch.tensor(expected, dtype=torch.long)
    config = TSPConfig(num_locations=4)
    instance = TSPSampler(config, seed=0)(tour.shape[0])
    solution = TSPSolution.create(instance=instance, tour=tour)

    neighbours_index = solution.neighbours_index()

    assert torch.equal(neighbours_index, expected)


@pytest.mark.parametrize(
    "tour,expected",
    [
        ([[0, 1, 2, 3]], [[0, 0, 0, 0]]),
        ([[0, 1, -1, 2]], [[0, 0, 0, -1]]),
        ([[-1, -1, -1, -1]], [[-1, -1, -1, -1]]),
    ],
)
def test_tour_index(tour, expected):
    tour = torch.tensor(tour, dtype=torch.long)
    expected = torch.tensor(expected, dtype=torch.long)
    config = TSPConfig(num_locations=4)
    instance = TSPSampler(config, seed=0)(tour.shape[0])
    solution = TSPSolution.create(instance=instance, tour=tour)

    tour_index = solution.tour_index()

    assert torch.equal(tour_index, expected)
