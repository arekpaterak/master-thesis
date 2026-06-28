import torch

from core.models.nds.baseline import (
    LeaveOneOutBaseline,
    LeaveOneOutBaselineConfig,
    NoBaseline,
    NoBaselineConfig,
    SharedBaseline,
    SharedBaselineConfig,
    build_baseline,
)


def test_min_rollouts():
    assert NoBaseline().min_rollouts() == 1
    assert SharedBaseline().min_rollouts() == 2
    assert LeaveOneOutBaseline().min_rollouts() == 2


def test_build_baseline_returns_expected_type():
    assert isinstance(build_baseline(NoBaselineConfig()), NoBaseline)
    assert isinstance(build_baseline(SharedBaselineConfig()), SharedBaseline)
    assert isinstance(build_baseline(LeaveOneOutBaselineConfig()), LeaveOneOutBaseline)


def test_no_baseline_is_zero():
    reward = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.equal(NoBaseline()(reward), torch.zeros_like(reward))


def test_shared_baseline_is_mean():
    reward = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.allclose(SharedBaseline()(reward), reward.mean(dim=1, keepdim=True))


def test_leave_one_out_matches_formula():
    reward = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.5, 0.5]])
    expected = (reward.sum(dim=1, keepdim=True) - reward) / (reward.shape[1] - 1)
    assert torch.allclose(LeaveOneOutBaseline()(reward), expected)


def test_leave_one_out_single_rollout_has_zero_advantage():
    reward = torch.tensor([[2.0], [3.0]])
    # baseline == reward, so advantage (reward - baseline) collapses to zero.
    assert torch.equal(LeaveOneOutBaseline()(reward), reward)
