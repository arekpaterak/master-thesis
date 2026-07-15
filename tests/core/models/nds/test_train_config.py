import pytest

from core.models.nds.config import JointModelConfig, TrainConfig


def test_original_pretrained_accepts_explicit_matching_encoder():
    model = JointModelConfig.model_validate(
        {
            "original_pretrained_weights": {"path": "checkpoint.pt"},
            "encoder": {"encoding_order": "route_neighbours", "route_aggregation": "sum"},
        }
    )
    assert model.encoder.route_aggregation == "sum"


def test_original_pretrained_rejects_conflicting_encoder():
    with pytest.raises(ValueError, match="route_aggregation"):
        JointModelConfig.model_validate(
            {
                "original_pretrained_weights": {"path": "checkpoint.pt"},
                "encoder": {"encoding_order": "route_neighbours", "route_aggregation": "mean"},
            }
        )


def test_mean_baseline_requires_at_least_two_rollouts():
    with pytest.raises(ValueError, match="num_rollouts >= 2"):
        TrainConfig.model_validate({"baseline": {"type": "mean"}, "search": {"num_rollouts": 1}})


def test_leave_one_out_baseline_requires_at_least_two_rollouts():
    with pytest.raises(ValueError, match="num_rollouts >= 2"):
        TrainConfig.model_validate(
            {"baseline": {"type": "leave_one_out"}, "search": {"num_rollouts": 1}}
        )
