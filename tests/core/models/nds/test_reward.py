import torch

from core.models.nds.reward import (
    AbsoluteReward,
    AbsoluteRewardConfig,
    RelativeReward,
    RelativeRewardConfig,
    build_reward,
)


def _objectives():
    objective_before = torch.tensor([10.0, 4.0])
    objective_after = torch.tensor([[8.0, 12.0, 10.0], [3.0, 4.0, 5.0]])
    return objective_before, objective_after


def test_build_reward_returns_expected_type():
    assert isinstance(build_reward(AbsoluteRewardConfig()), AbsoluteReward)
    assert isinstance(build_reward(RelativeRewardConfig()), RelativeReward)


def test_absolute_reward_is_clamped_objective_delta():
    before, after = _objectives()

    reward = AbsoluteReward()(objective_before=before, objective_after=after)

    expected = (before.unsqueeze(-1) - after).clamp(min=0.0)
    assert torch.allclose(reward, expected)
    # Worse-or-equal rollouts earn no reward.
    assert torch.equal(reward[0, 1:], torch.zeros(2))


def test_relative_reward_is_fraction_of_current_objective():
    before, after = _objectives()

    reward = RelativeReward()(objective_before=before, objective_after=after)

    expected = ((before.unsqueeze(-1) - after) / before.unsqueeze(-1)).clamp(min=0.0)
    assert torch.allclose(reward, expected)
    # 10 -> 8 is a 20% improvement; 4 -> 3 is a 25% improvement.
    assert torch.allclose(reward[:, 0], torch.tensor([0.2, 0.25]))


def test_relative_reward_is_scale_invariant():
    before, after = _objectives()

    reward = RelativeReward()(objective_before=before, objective_after=after)
    scaled = RelativeReward()(objective_before=before * 100, objective_after=after * 100)

    assert torch.allclose(reward, scaled)
