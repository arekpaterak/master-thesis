import pytest
import torch

from core.envs.destroy import DestroyEnv
from core.problems.cmstp import CMSTPDistribution, CMSTPSampler
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.tsp import TSPDistribution, TSPSampler


def _tsp(batch_size: int):
    return TSPSampler(TSPDistribution(num_locations=6), seed=0)(batch_size)


def _cvrp(batch_size: int):
    return CVRPSampler(CVRPDistribution.for_n(10), seed=0)(batch_size)


def _cmstp(batch_size: int):
    return CMSTPSampler(CMSTPDistribution.for_n(10), seed=0)(batch_size)


@pytest.mark.parametrize(("make_instance", "num_fixed"), [(_tsp, 0), (_cvrp, 1), (_cmstp, 1)])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_reset_forbids_only_fixed_nodes(make_instance, num_fixed, batch_size):
    instance = make_instance(batch_size)
    state = DestroyEnv(horizon=2).reset(instance)

    assert state.action_mask.shape == (batch_size, 1, instance.num_nodes)
    assert state.action_mask[..., :num_fixed].all()
    assert not state.action_mask[..., num_fixed:].any()
    assert state.last_node_idx is None


def test_step_forbids_chosen_nodes_per_rollout():
    env = DestroyEnv(horizon=2)
    state = env.reset(_cvrp(2))

    state = env.step(state, torch.tensor([[3, 4, 5], [1, 2, 3]]))
    state = env.step(state, torch.tensor([[6, 7, 8], [4, 5, 6]]))

    assert state.action_mask.shape == (2, 3, 11)
    assert state.action_mask[0, 0, [0, 3, 6]].all()
    assert state.action_mask.sum(-1).eq(3).all()
    assert torch.equal(state.last_node_idx.squeeze(-1), torch.tensor([[6, 7, 8], [4, 5, 6]]))


def test_done_after_horizon_steps():
    env = DestroyEnv(horizon=2)
    state = env.reset(_tsp(1))
    actions = torch.tensor([[[1, 2]]])

    assert not env.all_done(state)
    assert not env.all_done(env.step(state, actions[..., 0]))
    assert env.all_done(env.replay(_tsp(1), actions))
