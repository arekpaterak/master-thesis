import torch

from core.problems.tsp import TSPInstance
from core.problems.tsp.constructive.env import TSPConstructiveEnv

ENV = TSPConstructiveEnv()


def _instance(batch_size: int, num_nodes: int) -> TSPInstance:
    return TSPInstance.create(locations=torch.rand(batch_size, num_nodes, 2))


def test_reset() -> None:
    state = ENV.reset(_instance(2, 5))

    assert ENV.action_mask(state).shape == (2, 5)
    assert not ENV.action_mask(state).any()
    assert state.is_initial
    assert not ENV.is_done(state).any()


def test_step_records_first_and_last_node() -> None:
    state = ENV.reset(_instance(1, 6))

    state = ENV.step(ENV.step(state, torch.tensor([4])), torch.tensor([2]))

    assert torch.equal(state.first_node_idx, torch.tensor([[4]]))
    assert torch.equal(state.last_node_idx, torch.tensor([[2]]))
    assert ENV.action_mask(state)[0, [2, 4]].all()
    assert ENV.action_mask(state).sum() == 2


def test_replay_partial_tour() -> None:
    partial_tour = torch.tensor([[1, 3], [2, 4]])
    state = ENV.replay(_instance(2, 5), partial_tour)

    assert torch.equal(state.first_node_idx, partial_tour[:, :1])
    assert torch.equal(state.last_node_idx, partial_tour[:, -1:])
    assert ENV.action_mask(state).gather(-1, partial_tour).all()


def test_done_after_visiting_every_node() -> None:
    instance = _instance(2, 4)
    tours = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    state = ENV.replay(instance, tours)

    assert ENV.is_done(state).all()
    assert torch.allclose(
        ENV.reward(instance, tours), -ENV.to_solution(instance, tours).objective()
    )
