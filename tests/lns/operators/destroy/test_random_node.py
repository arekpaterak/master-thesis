import pytest

from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.problems.cvrp import CVRPConfig, CVRPSampler, CVRPSolution
from core.problems.cvrp.solve import one_customer_per_route
from core.problems.solution import BaseSolution
from core.problems.tsp import TSPConfig, TSPSampler
from core.problems.tsp.solution import TSPSolution
from core.problems.tsp.solve import random_permutation


@pytest.fixture
def tsp_solution() -> TSPSolution:
    batch_size = 3
    n = 10
    instance = TSPSampler(TSPConfig(num_locations=n))(batch_size)
    return random_permutation(instance)


@pytest.fixture
def cvrp_solution() -> CVRPSolution:
    batch_size = 3
    n = 10
    instance = CVRPSampler(CVRPConfig.without_capacity(n))(batch_size)
    solution = one_customer_per_route(instance)
    return solution


@pytest.mark.parametrize("destruction_size", [-1])
def test_random_node_should_fail_on_creation(destruction_size: int):
    with pytest.raises(ValueError):
        RandomNodeDestroy(destruction_size=destruction_size, seed=42)


@pytest.mark.parametrize(
    "solution_fixture,destruction_size",
    [
        ("tsp_solution", 0),
        ("tsp_solution", 1),
        ("tsp_solution", 5),
        ("tsp_solution", 10),
        ("cvrp_solution", 0),
        ("cvrp_solution", 1),
        ("cvrp_solution", 5),
        ("cvrp_solution", 10),
    ],
)
def test_random_node(solution_fixture: str, destruction_size: int, request):
    solution = request.getfixturevalue(solution_fixture)
    destroy = RandomNodeDestroy(destruction_size=destruction_size, seed=42)
    destroyed: BaseSolution = destroy(solution)
    actual_destruction_size = destroyed.num_unassigned
    assert (actual_destruction_size == destruction_size).all()


@pytest.mark.parametrize(
    "solution_fixture,destruction_size",
    [
        ("tsp_solution", 20),
        ("cvrp_solution", 11),
        ("cvrp_solution", 20),
    ],
)
def test_random_node_fail_on_too_large_destruction_size(
    solution_fixture: str, destruction_size: int, request
):
    solution = request.getfixturevalue(solution_fixture)
    destroy = RandomNodeDestroy(destruction_size=destruction_size, seed=42)
    with pytest.raises(ValueError):
        destroy(solution)
