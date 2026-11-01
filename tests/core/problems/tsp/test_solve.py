import pytest

from core.problems.tsp import TSPDistribution, TSPInstance, TSPSampler
from core.problems.tsp.solution import TSPSolution
from core.problems.tsp.solve import STRATEGIES


@pytest.fixture
def simple_instance() -> TSPInstance:
    config = TSPDistribution(num_locations=10, distribution_of_locations="uniform")
    sampler = TSPSampler(config)
    batch_size = 5
    return sampler(batch_size)


@pytest.fixture
def sampled_instance(request) -> TSPInstance:
    batch_size, num_locations = request.param
    config = TSPDistribution(
        num_locations=num_locations,
        distribution_of_locations="uniform",
    )
    sampler = TSPSampler(config)
    return sampler(batch_size)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategy_returns_solution(strategy, simple_instance):
    solution = strategy(simple_instance)

    assert isinstance(solution, TSPSolution)
    assert solution.batch_size == simple_instance.batch_size


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_solution_shape(strategy, simple_instance):
    batch_size, num_locations, _ = simple_instance.locations.shape

    solution = strategy(simple_instance)

    assert solution.tour.shape == (batch_size, num_locations)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_solutions_are_valid_permutations(strategy, simple_instance):
    num_locations = simple_instance.locations.shape[1]
    expected = set(range(num_locations))

    solution = strategy(simple_instance)

    for tour in solution.tour:
        assert set(tour.tolist()) == expected


@pytest.mark.parametrize(
    "sampled_instance",
    [
        (1, 5),
        (3, 10),
        (8, 20),
    ],
    indirect=True,
    ids=lambda x: f"{x[0]}x{x[1]}",
)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_various_instance_sizes(strategy, sampled_instance: TSPInstance):
    batch_size, num_locations, _ = sampled_instance.locations.shape

    solution = strategy(sampled_instance)

    assert solution.tour.shape == (batch_size, num_locations)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_without_batch_dim(strategy, simple_instance):
    single_instance = simple_instance[0]
    assert single_instance.batch_dims == 0
    num_locations = single_instance.num_locations

    solution = strategy(single_instance)

    assert solution.batch_dims == 0
    assert solution.tour.shape == (num_locations,)
