import pytest

from core.models.nds.baseline import (
    GreedyRolloutBaselineConfig,
    SharedBaselineConfig,
)
from core.models.nds.config import TrainConfig


def test_default_baseline_is_shared_mean():
    assert isinstance(TrainConfig().baseline, SharedBaselineConfig)


def test_mean_baseline_requires_at_least_two_rollouts():
    with pytest.raises(ValueError, match="num_rollouts >= 2"):
        TrainConfig.model_validate({"baseline": {"type": "mean"}, "search": {"num_rollouts": 1}})


def test_leave_one_out_baseline_requires_at_least_two_rollouts():
    with pytest.raises(ValueError, match="num_rollouts >= 2"):
        TrainConfig.model_validate(
            {"baseline": {"type": "leave_one_out"}, "search": {"num_rollouts": 1}}
        )


def _greedy_rollout_config(**overrides) -> dict:
    base = {
        "baseline": {"type": "greedy_rollout"},
        "search": {"num_rollouts": 1},
        "model": {"decoder": {"type": "simple", "with_poly": False}},
    }
    base.update(overrides)
    return base


def test_greedy_rollout_baseline_single_rollout_without_poly_is_valid():
    config = TrainConfig.model_validate(_greedy_rollout_config())
    assert isinstance(config.baseline, GreedyRolloutBaselineConfig)
    assert config.search.num_rollouts == 1
    assert config.model.decoder.with_poly is False


def test_greedy_rollout_baseline_rejects_multiple_rollouts():
    with pytest.raises(ValueError, match="num_rollouts == 1"):
        TrainConfig.model_validate(_greedy_rollout_config(search={"num_rollouts": 8}))


def test_greedy_rollout_baseline_requires_poly_disabled():
    with pytest.raises(ValueError, match="with_poly = false"):
        TrainConfig.model_validate(
            _greedy_rollout_config(model={"decoder": {"type": "simple", "with_poly": True}})
        )
