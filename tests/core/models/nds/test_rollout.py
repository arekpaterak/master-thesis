import pytest

from core.models.nds.model import NDSModel
from core.problems.cvrp import CVRPConfig, CVRPSampler, CVRPSolution
from core.problems.cvrp.solve import one_customer_per_route


@pytest.fixture(scope="session")
def nds_test_model() -> NDSModel:
    return NDSModel(problem_name="cvrp")


@pytest.fixture
def example_solution() -> CVRPSolution:
    sampler = CVRPSampler(CVRPConfig.for_n(20))
    instance = sampler(4)
    solution = one_customer_per_route(instance)
    return solution


def test_if_can_produce_rollout(
    nds_test_model: NDSModel,
    example_solution: CVRPSolution,
):
    ROLLOUTS_PER_SOLUTION = 10
    DECONSTRUCTION_SIZE = 15

    rollout = nds_test_model.rollout(
        example_solution,
        rollouts_per_solution=ROLLOUTS_PER_SOLUTION,
        deconstruction_size=DECONSTRUCTION_SIZE,
        greedy=False,
    )

    assert rollout.actions.shape == (4, ROLLOUTS_PER_SOLUTION, DECONSTRUCTION_SIZE)
    assert rollout.log_p.shape == (4, ROLLOUTS_PER_SOLUTION)
    for action_sequence in rollout.actions.reshape(-1, DECONSTRUCTION_SIZE):
        assert len(set(action_sequence)) == DECONSTRUCTION_SIZE, (
            "Expected no duplicate actions in a rollout."
        )
