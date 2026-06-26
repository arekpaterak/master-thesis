import torch

from core.models.nds.policy_loss import (
    AllRolloutsLoss,
    AllRolloutsLossConfig,
    BestRolloutLoss,
    BestRolloutLossConfig,
    build_policy_loss,
)


def _reward_log_p():
    reward = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.5, 0.5]])
    log_p = torch.tensor([[-1.0, -2.0, -0.5], [-0.2, -1.5, -3.0]], requires_grad=True)
    return reward, log_p


def test_build_policy_loss_returns_expected_type():
    assert isinstance(build_policy_loss(BestRolloutLossConfig()), BestRolloutLoss)
    assert isinstance(build_policy_loss(AllRolloutsLossConfig()), AllRolloutsLoss)


def test_best_rollout_loss_matches_legacy_formula():
    reward, log_p = _reward_log_p()

    loss = BestRolloutLoss()(reward=reward, log_p=log_p)

    baseline = reward.mean(dim=1, keepdim=True)
    best_idx = reward.argmax(dim=1, keepdim=True)
    best_reward = reward.gather(1, best_idx)
    best_log_p = log_p.gather(1, best_idx)
    expected = -((best_reward - baseline) * best_log_p).mean()
    assert torch.allclose(loss, expected)


def test_best_rollout_loss_only_grads_best_entry():
    reward, log_p = _reward_log_p()

    BestRolloutLoss()(reward=reward, log_p=log_p).backward()

    grad_nonzero = log_p.grad != 0
    expected = torch.zeros_like(grad_nonzero)
    expected.scatter_(1, reward.argmax(dim=1, keepdim=True), True)
    assert torch.equal(grad_nonzero, expected)


def test_all_rollouts_mean_baseline_matches_formula():
    reward, log_p = _reward_log_p()

    loss = AllRolloutsLoss(baseline="mean")(reward=reward, log_p=log_p)

    baseline = reward.mean(dim=1, keepdim=True)
    expected = -((reward - baseline) * log_p).mean()
    assert torch.allclose(loss, expected)


def test_all_rollouts_grads_every_entry():
    reward, log_p = _reward_log_p()

    AllRolloutsLoss(baseline="mean")(reward=reward, log_p=log_p).backward()

    assert torch.all(log_p.grad != 0)


def test_all_rollouts_leave_one_out_matches_formula():
    reward, log_p = _reward_log_p()

    loss = AllRolloutsLoss(baseline="leave_one_out")(reward=reward, log_p=log_p)

    num_rollouts = reward.shape[1]
    baseline = (reward.sum(dim=1, keepdim=True) - reward) / (num_rollouts - 1)
    expected = -((reward - baseline) * log_p).mean()
    assert torch.allclose(loss, expected)


def test_leave_one_out_single_rollout_is_finite_and_zero_advantage():
    reward = torch.tensor([[2.0], [3.0]])
    log_p = torch.tensor([[-1.0], [-0.5]], requires_grad=True)

    loss = AllRolloutsLoss(baseline="leave_one_out")(reward=reward, log_p=log_p)

    # With a single rollout the advantage collapses to zero (baseline == reward).
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.zeros(()))
