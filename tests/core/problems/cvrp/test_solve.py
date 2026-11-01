import pytest

from core.problems.cvrp import CVRPDistribution, CVRPInstance, CVRPSampler
from core.problems.cvrp.solution import CVRPSolution
from core.problems.cvrp.solve import STRATEGIES


@pytest.fixture
def simple_instance() -> CVRPInstance:
    config = CVRPDistribution.for_n(20)
    sampler = CVRPSampler(config)
    batch_size = 5
    return sampler(batch_size)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategy_returns_solution(strategy, simple_instance):
    solution = strategy(simple_instance)

    assert isinstance(solution, CVRPSolution)
    assert solution.batch_size == simple_instance.batch_size


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_solution_contains_all_customers(strategy, simple_instance):
    """Every customer must appear exactly once in the sequences."""
    num_customers = simple_instance.num_customers
    expected_customers = set(range(1, num_customers + 1))

    solution = strategy(simple_instance)

    sequences = solution.sequence if solution.batch_dims > 0 else [solution.sequence]

    for seq in sequences:
        customers_in_tour = seq[seq != 0].tolist()
        assert set(customers_in_tour) == expected_customers
        assert len(customers_in_tour) == num_customers


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_without_batch_dim(strategy, simple_instance):
    single_instance = simple_instance[0]
    assert single_instance.batch_dims == 0

    solution = strategy(single_instance)

    assert solution.batch_dims == 0
    assert solution.sequence.ndim == 1
