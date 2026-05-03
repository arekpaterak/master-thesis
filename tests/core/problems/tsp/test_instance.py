import pytest

from core.problems.tsp import TSPConfig, TSPSampler


def test_if_can_visualize_without_batch_dim():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)[0]
    assert instance.batch_dims == 0

    instance.visualize()


def test_if_can_visualize_with_batch_of_one():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)[:1]
    assert instance.batch_dims == 1

    instance.visualize()


def test_if_fails_to_visualize_batch():
    instance = TSPSampler(TSPConfig(num_locations=10))(10)

    with pytest.raises(ValueError):
        instance.visualize()
