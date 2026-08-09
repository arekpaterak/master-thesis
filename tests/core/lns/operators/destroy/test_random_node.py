import pytest

from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.problems.cmstp import CMSTPSolution
from core.problems.cmstp.config import CMSTP
from core.problems.cmstp.sampler import CMSTPSampler
from core.problems.cvrp import CVRP, CVRPSampler, CVRPSolution
from core.problems.cvrp.solve import one_customer_per_route
from core.problems.solution import BaseSolution
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.solution import TSPSolution
from core.problems.tsp.solve import random_permutation


@pytest.fixture
def tsp_solution() -> TSPSolution:
    batch_size = 3
    n = 10
    instance = TSPSampler(TSP(num_locations=n))(batch_size)
    return random_permutation(instance)


@pytest.fixture
def cvrp_solution() -> CVRPSolution:
    batch_size = 3
    n = 10
    instance = CVRPSampler(CVRP.without_capacity(n))(batch_size)
    solution = one_customer_per_route(instance)
    return solution


@pytest.fixture
def cmstp_solution() -> CMSTPSolution:
    batch_size = 3
    n = 10
    instance = CMSTPSampler(CMSTP(num_customers=n, capacity_raw=30), seed=0)(batch_size)
    return CMSTPSolution.star(instance)


@pytest.mark.parametrize("destruction_size", [-1])
def test_random_node_should_fail_on_creation_with_bad_size(destruction_size: int):
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


# CMSTP has no non-numba remove_nodes, so the warmup only calls select_nodes; test that directly.
@pytest.mark.parametrize("destruction_size", [0, 1, 5, 10])
def test_random_node_select_nodes_cmstp(cmstp_solution: CMSTPSolution, destruction_size: int):
    num_rollouts = 4
    num_customers = cmstp_solution.instance.num_customers
    destroy = RandomNodeDestroy(destruction_size=destruction_size, seed=42)

    nodes = destroy.select(cmstp_solution, num_rollouts).nodes

    assert nodes.shape == (*cmstp_solution.batch_size, num_rollouts, destruction_size)
    # The depot (index 0) is never removed and every rollout picks distinct customers.
    assert (nodes >= 1).all() and (nodes <= num_customers).all()
    for per_rollout in nodes.flatten(end_dim=-2).tolist():
        assert len(set(per_rollout)) == destruction_size


@pytest.mark.parametrize("destruction_size", [11, 20])
def test_random_node_select_nodes_cmstp_fail_on_too_large_destruction_size(
    cmstp_solution: CMSTPSolution, destruction_size: int
):
    destroy = RandomNodeDestroy(destruction_size=destruction_size, seed=42)
    with pytest.raises(ValueError):
        destroy.select(cmstp_solution, num_rollouts=4)
