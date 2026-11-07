import torch

from core.problems.cvrp import CVRPInstance
from core.problems.cvrp.constructive.env import CVRPConstructiveEnv

ENV = CVRPConstructiveEnv()


def _instance() -> CVRPInstance:
    return CVRPInstance.create(
        locations=torch.rand(2, 4, 2),
        demand=torch.tensor([[0.0, 4, 4, 4], [0.0, 2, 2, 2]]),
        capacity=torch.full((2, 1), 10.0),
    )


def test_reset_starts_at_depot_with_full_vehicle() -> None:
    state = ENV.reset(_instance())

    assert torch.equal(state.last_node_idx, torch.zeros(2, 1, dtype=torch.long))
    assert torch.equal(state.remaining_capacity, torch.ones(2, 1))
    assert torch.equal(ENV.action_mask(state)[:, 0], torch.tensor([True, True]))
    assert not ENV.action_mask(state)[:, 1:].any()


def test_customers_over_capacity_are_masked() -> None:
    state = ENV.replay(_instance(), torch.tensor([[1, 2], [1, 2]]))

    assert torch.allclose(state.remaining_capacity, torch.tensor([[0.2], [0.6]]))
    assert torch.equal(ENV.action_mask(state)[0], torch.tensor([False, True, True, True]))
    assert torch.equal(ENV.action_mask(state)[1], torch.tensor([False, True, True, False]))


def test_done_only_back_at_depot_after_serving_everyone() -> None:
    instance = _instance()
    served = ENV.replay(instance, torch.tensor([[1, 2, 0, 3], [1, 2, 3, 3]]))
    assert not ENV.is_done(served).any()

    sequence = torch.tensor([[1, 2, 0, 3, 0], [1, 2, 3, 0, 0]])
    done = ENV.replay(instance, sequence)

    assert ENV.is_done(done).all()
    assert torch.equal(ENV.action_mask(done)[:, 0], torch.tensor([False, False]))
    assert torch.allclose(
        ENV.reward(instance, sequence), -ENV.to_solution(instance, sequence).objective()
    )
