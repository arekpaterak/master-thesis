import pytest

from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route


def test_if_can_visualize_without_batch_dim():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)[0]
    solution = one_customer_per_route(instance)
    assert solution.batch_dims == 0

    solution.visualize()


def test_if_can_visualize_with_batch_of_one():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)[:1]
    solution = one_customer_per_route(instance)
    assert solution.batch_dims == 1

    solution.visualize()


def test_if_fails_to_visualize_batch():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)

    with pytest.raises(ValueError):
        solution.visualize()
