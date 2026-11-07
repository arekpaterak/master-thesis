import pytest

from core.models.nds.config import JointModelConfig, TrainConfig


def test_original_pretrained_accepts_explicit_matching_encoder():
    model = JointModelConfig.model_validate(
        {
            "problem_name": "cvrp",
            "original_pretrained_weights": {"path": "checkpoint.pt"},
            "encoder": {"encoding_order": "route_neighbours", "route_aggregation": "sum"},
        }
    )
    assert model.encoder.route_aggregation == "sum"


def test_original_pretrained_rejects_conflicting_encoder():
    with pytest.raises(ValueError, match="route_aggregation"):
        JointModelConfig.model_validate(
            {
                "problem_name": "cvrp",
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


def test_slot_decoder_needs_a_slot_per_removed_node():
    with pytest.raises(ValueError, match="selects at most 5 nodes, but the search removes 10"):
        TrainConfig.model_validate(
            {
                "model": {"decoder": {"type": "slot", "num_slots": 5}},
                "search": {"deconstruction_size": 10},
                "validation": {"search": {"deconstruction_size": 5}},
            }
        )


def test_slot_decoder_may_have_more_slots_than_removed_nodes():
    config = TrainConfig.model_validate(
        {
            "model": {"decoder": {"type": "slot", "num_slots": 15}},
            "search": {"deconstruction_size": 10},
            "validation": {"search": {"deconstruction_size": 10}},
        }
    )
    assert config.model.decoder.max_deconstruction_size == 15
