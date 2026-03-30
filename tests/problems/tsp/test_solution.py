import pytest

from core.problems.tsp import TSPConfig, TSPSampler
from core.problems.tsp.solve import random_permutation


def test_if_can_visualize_without_batch_dim():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)[0]
    solution = random_permutation(instance)
    assert solution.batch_dims == 0

    solution.visualize()


def test_if_can_visualize_with_batch_of_one():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)[:1]
    solution = random_permutation(instance)
    assert solution.batch_dims == 1

    solution.visualize()


def test_if_fails_to_visualize_batch():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)
    solution = random_permutation(instance)

    with pytest.raises(ValueError):
        solution.visualize()


def test_objective():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)
    solution = random_permutation(instance)

    assert (solution.objective() > 0).all()


def test_deconstructed():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)
    solution = random_permutation(instance)

    solution[0, 1] = -1

    assert (solution.is_complete == False)[0]
    assert (solution.is_complete == True)[1:].all()

    solution[:, 1] = -1

    assert (solution.is_complete == False).all()
