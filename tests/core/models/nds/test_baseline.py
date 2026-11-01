import torch

from core.models.nds.baseline import (
    LeaveOneOutBaseline,
    LeaveOneOutBaselineConfig,
    NoBaseline,
    NoBaselineConfig,
    SharedBaseline,
    SharedBaselineConfig,
)


def test_min_rollouts():
    assert NoBaselineConfig().min_rollouts == 1
    assert SharedBaselineConfig().min_rollouts == 2
    assert LeaveOneOutBaselineConfig().min_rollouts == 2


def test_min_rollouts_is_not_serialized():
    assert "min_rollouts" not in SharedBaselineConfig().model_dump()


def test_config_builds_expected_type():
    assert isinstance(NoBaselineConfig().build(), NoBaseline)
    assert isinstance(SharedBaselineConfig().build(), SharedBaseline)
    assert isinstance(LeaveOneOutBaselineConfig().build(), LeaveOneOutBaseline)


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
