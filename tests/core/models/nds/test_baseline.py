import torch
import torch.nn as nn

from core.models.nds import baseline as baseline_module
from core.models.nds.baseline import (
    GreedyRolloutBaseline,
    GreedyRolloutBaselineConfig,
    LeaveOneOutBaseline,
    LeaveOneOutBaselineConfig,
    NoBaseline,
    NoBaselineConfig,
    SharedBaseline,
    SharedBaselineConfig,
    build_baseline,
)
from core.models.statistical_test import OneSidedPairedTTestResult


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3))


def _improved(value: bool):
    return lambda **_: OneSidedPairedTTestResult(
        improved=value, model_mean=1.0, baseline_mean=2.0, p_value=0.0
    )


def test_min_rollouts():
    assert NoBaseline().min_rollouts() == 1
    assert SharedBaseline().min_rollouts() == 2
    assert LeaveOneOutBaseline().min_rollouts() == 2
    assert GreedyRolloutBaseline().min_rollouts() == 1


def test_build_baseline_returns_expected_type():
    assert isinstance(build_baseline(NoBaselineConfig()), NoBaseline)
    assert isinstance(build_baseline(SharedBaselineConfig()), SharedBaseline)
    assert isinstance(build_baseline(LeaveOneOutBaselineConfig()), LeaveOneOutBaseline)
    assert isinstance(build_baseline(GreedyRolloutBaselineConfig()), GreedyRolloutBaseline)


def test_no_baseline_is_zero():
    reward = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = NoBaseline()(reward=reward, context=None)
    assert torch.equal(out, torch.zeros_like(reward))


def test_shared_baseline_is_mean():
    reward = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = SharedBaseline()(reward=reward, context=None)
    assert torch.allclose(out, reward.mean(dim=1, keepdim=True))


def test_leave_one_out_matches_formula():
    reward = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.5, 0.5]])
    out = LeaveOneOutBaseline()(reward=reward, context=None)
    expected = (reward.sum(dim=1, keepdim=True) - reward) / (reward.shape[1] - 1)
    assert torch.allclose(out, expected)


def test_leave_one_out_single_rollout_has_zero_advantage():
    reward = torch.tensor([[2.0], [3.0]])
    out = LeaveOneOutBaseline()(reward=reward, context=None)
    # baseline == reward, so advantage (reward - baseline) collapses to zero.
    assert torch.equal(out, reward)


def test_greedy_baseline_setup_freezes_copy():
    policy = _TinyPolicy()
    baseline = GreedyRolloutBaseline()
    baseline.setup(policy)
    assert all(not p.requires_grad for p in baseline._baseline_model.parameters())


def test_greedy_baseline_syncs_when_improved(monkeypatch):
    policy = _TinyPolicy()
    baseline = GreedyRolloutBaseline(sync_every_n_epochs=1)
    baseline.setup(policy)
    with torch.no_grad():
        policy.w.add_(1.0)  # diverge the policy from the baseline copy
    monkeypatch.setattr(baseline_module, "one_sided_paired_ttest", _improved(True))

    result = baseline.epoch_callback(policy=policy, metric=lambda _: torch.zeros(2), epoch=0)

    assert result is not None and result.improved
    assert torch.equal(baseline._baseline_model.w, policy.w)


def test_greedy_baseline_does_not_sync_when_not_improved(monkeypatch):
    policy = _TinyPolicy()
    baseline = GreedyRolloutBaseline(sync_every_n_epochs=1)
    baseline.setup(policy)
    with torch.no_grad():
        policy.w.add_(1.0)
    monkeypatch.setattr(baseline_module, "one_sided_paired_ttest", _improved(False))

    baseline.epoch_callback(policy=policy, metric=lambda _: torch.zeros(2), epoch=0)

    assert torch.equal(baseline._baseline_model.w, torch.zeros(3))


def test_greedy_baseline_respects_sync_cadence(monkeypatch):
    policy = _TinyPolicy()
    baseline = GreedyRolloutBaseline(sync_every_n_epochs=5)
    baseline.setup(policy)
    calls: list[int] = []
    monkeypatch.setattr(
        baseline_module,
        "one_sided_paired_ttest",
        lambda **_: calls.append(1) or _improved(False)(),
    )

    assert baseline.epoch_callback(policy=policy, metric=lambda _: None, epoch=0) is None
    assert not calls
    baseline.epoch_callback(policy=policy, metric=lambda _: None, epoch=4)
    assert calls == [1]


def test_greedy_baseline_state_dict_roundtrip():
    policy = _TinyPolicy()
    baseline = GreedyRolloutBaseline()
    baseline.setup(policy)
    with torch.no_grad():
        baseline._baseline_model.w.add_(2.0)
    state = baseline.state_dict()

    restored = GreedyRolloutBaseline()
    restored.setup(_TinyPolicy())
    restored.load_state_dict(state)

    assert torch.equal(restored._baseline_model.w, baseline._baseline_model.w)
