from pathlib import Path

import pytest
import torch

from core.models.nds.decoders.ar import AutoregressiveNDSDecoder
from core.models.nds.encoders.config import EncoderConfig
from core.models.nds.encoders.nds import NDSEncoder
from core.models.nds.models.joint import JointNDSModel
from core.models.nds.pretrained.load import (
    load_original_pretrained_decoder_weights,
    load_original_pretrained_encoder_weights,
)
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.features import (
    CVRPInstanceFeatureExtractor,
    CVRPSolutionFeatureExtractor,
)
from core.problems.cvrp.solve import one_customer_per_route

CHECKPOINT = Path("../experiments/pretrained/nds/cvrp_100/checkpoint-2000.pt")

PRETRAINED_ENCODER_CONFIG = EncoderConfig(
    encoding_order="route_neighbours", route_aggregation="sum"
)

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"Pretrained checkpoint not found: {CHECKPOINT}",
)


@requires_checkpoint
def test_load_pretrained_changes_weights() -> None:
    encoder = NDSEncoder(problem_name="cvrp", encoder_config=PRETRAINED_ENCODER_CONFIG)
    before = encoder.pre_attention[0].MHA.in_proj_weight.clone()

    load_original_pretrained_encoder_weights(encoder, CHECKPOINT)

    after = encoder.pre_attention[0].MHA.in_proj_weight
    assert not torch.equal(before, after), (
        "Weights should change after loading pretrained checkpoint"
    )


@requires_checkpoint
def test_load_pretrained_zeros_in_proj_bias() -> None:
    encoder = NDSEncoder(problem_name="cvrp", encoder_config=PRETRAINED_ENCODER_CONFIG)
    load_original_pretrained_encoder_weights(encoder, CHECKPOINT)

    for i, layer in enumerate((*encoder.pre_attention, *encoder.post_attention)):
        assert torch.all(layer.MHA.in_proj_bias == 0), (
            f"in_proj_bias should be zero for attention layer {i} (NDS has no Q/K/V bias)"
        )


@requires_checkpoint
def test_load_pretrained_encoder_forward() -> None:
    B, N = 2, 100
    sampler = CVRPSampler(CVRP.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)

    encoder = NDSEncoder(problem_name="cvrp", encoder_config=PRETRAINED_ENCODER_CONFIG)
    load_original_pretrained_encoder_weights(encoder, CHECKPOINT)
    encoder.eval()

    instance_features = CVRPInstanceFeatureExtractor()(instance)
    solution_features = CVRPSolutionFeatureExtractor()(solution)
    with torch.no_grad():
        output = encoder(instance_features, solution_features)

    assert output.shape == (B, N + 1, 128)


@requires_checkpoint
def test_load_pretrained_decoder() -> None:
    decoder = AutoregressiveNDSDecoder(hidden_dim=128, seed_dim=10, poly_dim=256, num_heads=8)
    before = decoder.MHA.W_Q.weight.clone()

    load_original_pretrained_decoder_weights(decoder, CHECKPOINT)

    assert not torch.equal(before, decoder.MHA.W_Q.weight), "Weights should change after loading"
    assert torch.all(decoder.MHA.W_Q.bias == 0), "W_Q bias should be zero (NDS has no Q bias)"


@requires_checkpoint
def test_load_pretrained_fails_on_incompatible_architecture() -> None:
    encoder = NDSEncoder(
        problem_name="cvrp", hidden_dim=64, encoder_config=PRETRAINED_ENCODER_CONFIG
    )  # checkpoint expects hidden_dim=128
    with pytest.raises(RuntimeError):
        load_original_pretrained_encoder_weights(encoder, CHECKPOINT)


@requires_checkpoint
def test_load_pretrained_model_forward() -> None:
    B, R, N, k = 2, 4, 100, 15
    sampler = CVRPSampler(CVRP.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)

    model = JointNDSModel(problem_name="cvrp", encoder_config=PRETRAINED_ENCODER_CONFIG)
    load_original_pretrained_encoder_weights(model.encoder, CHECKPOINT)
    model.eval()

    with torch.no_grad():
        rollout = model.rollout(solution, rollouts_per_solution=R, deconstruction_size=k)

    assert rollout.actions.shape == (B, R, k)


def test_load_guard_rejects_mean_route_aggregation() -> None:
    encoder = NDSEncoder(
        problem_name="cvrp",
        encoder_config=EncoderConfig(encoding_order="route_neighbours", route_aggregation="mean"),
    )
    with pytest.raises(ValueError, match="route aggregation"):
        load_original_pretrained_encoder_weights(encoder, CHECKPOINT)


def test_load_guard_rejects_neighbours_first_order() -> None:
    encoder = NDSEncoder(
        problem_name="cvrp",
        encoder_config=EncoderConfig(encoding_order="neighbours_route", route_aggregation="sum"),
    )
    with pytest.raises(ValueError, match="encoding_order"):
        load_original_pretrained_encoder_weights(encoder, CHECKPOINT)
