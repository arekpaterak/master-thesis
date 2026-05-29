from pathlib import Path

import pytest
import torch

from core.models.nds.decoders.ar import AutoregressiveNDSDecoder
from core.models.nds.encoder import NDSEncoder
from core.models.nds.models.ar import AutoregressiveNDSModel
from core.models.nds.pretrained import (
    load_original_pretrained_decoder_weights,
    load_original_pretrained_encoder_weights,
)
from core.models.nds.seed_sampler import sample_seed
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.features import CVRPSolutionFeatureExtractor
from core.problems.cvrp.solve import one_customer_per_route

CHECKPOINT = Path("experiments/pretrained/nds/cvrp_100/checkpoint-2000.pt")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"Pretrained checkpoint not found: {CHECKPOINT}",
)


def test_load_pretrained_changes_weights() -> None:
    encoder = NDSEncoder(problem_name="cvrp")
    before = encoder.pre_attention[0].MHA.in_proj_weight.clone()

    load_original_pretrained_encoder_weights(encoder, CHECKPOINT)

    after = encoder.pre_attention[0].MHA.in_proj_weight
    assert not torch.equal(before, after), (
        "Weights should change after loading pretrained checkpoint"
    )


def test_load_pretrained_zeros_in_proj_bias() -> None:
    encoder = NDSEncoder(problem_name="cvrp")
    load_original_pretrained_encoder_weights(encoder, CHECKPOINT)

    for i, layer in enumerate((*encoder.pre_attention, *encoder.post_attention)):
        assert torch.all(layer.MHA.in_proj_bias == 0), (
            f"in_proj_bias should be zero for attention layer {i} (NDS has no Q/K/V bias)"
        )


def test_load_pretrained_encoder_forward() -> None:
    B, N = 2, 100
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)

    encoder = NDSEncoder(problem_name="cvrp")
    load_original_pretrained_encoder_weights(encoder, CHECKPOINT)
    encoder.eval()

    features = CVRPSolutionFeatureExtractor()(solution)
    with torch.no_grad():
        output = encoder(**features)

    assert output.shape == (B, N + 1, 128)


def test_load_pretrained_decoder() -> None:
    decoder = AutoregressiveNDSDecoder(hidden_dim=128, seed_dim=10, poly_dim=256, num_heads=8)
    before = decoder.MHA.W_Q.weight.clone()

    load_original_pretrained_decoder_weights(decoder, CHECKPOINT)

    assert not torch.equal(before, decoder.MHA.W_Q.weight), "Weights should change after loading"
    assert torch.all(decoder.MHA.W_Q.bias == 0), "W_Q bias should be zero (NDS has no Q bias)"


def test_load_pretrained_fails_on_incompatible_architecture() -> None:
    encoder = NDSEncoder(problem_name="cvrp", hidden_dim=64)  # checkpoint expects hidden_dim=128
    with pytest.raises(RuntimeError):
        load_original_pretrained_encoder_weights(encoder, CHECKPOINT)


def test_load_pretrained_model_forward() -> None:
    B, R, N = 2, 4, 100
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)

    model = AutoregressiveNDSModel(problem_name="cvrp")
    load_original_pretrained_encoder_weights(model.encoder, CHECKPOINT)
    model.eval()

    seed = sample_seed(batch_size=B, rollout_size=R, seed_dim=10)
    features = CVRPSolutionFeatureExtractor()(solution)

    with torch.no_grad():
        problem_state, decoder_state, cache = model.pre_forward(
            seed=seed, rollout_size=R, **features
        )
        log_p, _ = model.forward(
            problem_state=problem_state, decoder_state=decoder_state, cache=cache
        )

    assert log_p.shape == (B, R, N + 1)
