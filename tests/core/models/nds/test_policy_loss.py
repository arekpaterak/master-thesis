import torch

from core.models.nds.policy_loss import (
    AllRolloutsLoss,
    AllRolloutsLossConfig,
    BestRolloutLoss,
    BestRolloutLossConfig,
    build_policy_loss,
)


def _reward_baseline_log_p():
    reward = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.5, 0.5]])
    baseline = reward.mean(dim=1, keepdim=True)
    log_p = torch.tensor([[-1.0, -2.0, -0.5], [-0.2, -1.5, -3.0]], requires_grad=True)
    return reward, baseline, log_p


def test_build_policy_loss_returns_expected_type():
    assert isinstance(build_policy_loss(BestRolloutLossConfig()), BestRolloutLoss)
    assert isinstance(build_policy_loss(AllRolloutsLossConfig()), AllRolloutsLoss)


def test_best_rollout_loss_matches_formula():
    reward, baseline, log_p = _reward_baseline_log_p()

    loss = BestRolloutLoss()(reward=reward, baseline=baseline, log_p=log_p)

    advantage = reward - baseline
    best_idx = reward.argmax(dim=1, keepdim=True)
    best_advantage = advantage.gather(1, best_idx)
    best_log_p = log_p.gather(1, best_idx)
    expected = -(best_advantage * best_log_p).mean()
    assert torch.allclose(loss, expected)


def test_best_rollout_loss_only_grads_best_entry():
    reward, baseline, log_p = _reward_baseline_log_p()

    BestRolloutLoss()(reward=reward, baseline=baseline, log_p=log_p).backward()

    grad_nonzero = log_p.grad != 0
    expected = torch.zeros_like(grad_nonzero)
    expected.scatter_(1, reward.argmax(dim=1, keepdim=True), True)
    assert torch.equal(grad_nonzero, expected)


def test_all_rollouts_loss_matches_formula():
    reward, baseline, log_p = _reward_baseline_log_p()

    loss = AllRolloutsLoss()(reward=reward, baseline=baseline, log_p=log_p)

    expected = -((reward - baseline) * log_p).mean()
    assert torch.allclose(loss, expected)


def test_all_rollouts_grads_every_entry():
    reward, baseline, log_p = _reward_baseline_log_p()

    AllRolloutsLoss()(reward=reward, baseline=baseline, log_p=log_p).backward()

    assert torch.all(log_p.grad != 0)


def test_loss_accepts_broadcastable_single_column_baseline():
    # A greedy-rollout baseline produces a (B, 1) reference that broadcasts over rollouts.
    reward = torch.tensor([[1.0, 2.0], [0.0, 4.0]])
    baseline = torch.tensor([[0.5], [1.0]])
    log_p = torch.tensor([[-1.0, -0.5], [-2.0, -0.1]], requires_grad=True)

    loss = AllRolloutsLoss()(reward=reward, baseline=baseline, log_p=log_p)

    expected = -((reward - baseline) * log_p).mean()
    assert torch.allclose(loss, expected)
