import pytest

from core.models.nds.baseline import SharedBaselineConfig
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
