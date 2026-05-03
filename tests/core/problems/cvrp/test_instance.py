import pytest

from core.problems.cvrp import CVRPConfig, CVRPSampler


def test_if_can_visualize_without_batch_dim():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)[0]
    assert instance.batch_dims == 0

    instance.visualize()


def test_if_can_visualize_with_batch_of_one():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)[:1]
    assert instance.batch_dims == 1

    instance.visualize()


def test_if_fails_to_visualize_batch():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)

    with pytest.raises(ValueError):
        instance.visualize()
