import math

import pytest
import torch
from torch import nn

from core.models.layers.attention_scaling import (
    AttentionScalingConfig,
    scaled_multihead_attention,
)
from core.models.layers.cached_multihead_attention import CachedMultiheadAttention
from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.models.layers.multihead_attention import MultiheadAttentionWrapper
from core.models.nds.config import JointModelConfig, SplitModelConfig
from core.models.nds.decoders.config import (
    AutoregressiveDecoderConfig,
    SimpleOnePassDecoderConfig,
    SlotDecoderConfig,
)
from core.models.nds.encoders.config import JointEncoderConfig, SplitEncoderConfig
from core.models.nds.encoders.layers import AttentionLayer
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.registry import build_nds_model
from core.random import SeededGenerator

D, H = 16, 4


def test_factor() -> None:
    assert AttentionScalingConfig().factor(100) == pytest.approx(math.log(100))
    reference = AttentionScalingConfig(reference_size=100)
    assert reference.factor(100) == 1.0
    assert reference.factor(1000) == pytest.approx(1.5)


def test_reference_size_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        AttentionScalingConfig(reference_size=1)


def test_disabled_by_default() -> None:
    assert JointEncoderConfig().attention_scaling is None
    assert SplitEncoderConfig().attention_scaling is None
    assert AutoregressiveDecoderConfig().attention_scaling is None
    assert SimpleOnePassDecoderConfig().attention_scaling is None


def test_unit_scale_matches_multihead_attention() -> None:
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(D, H, batch_first=True).eval()
    q, kv = torch.randn(3, 2, D), torch.randn(3, 7, D)
    attend = torch.rand(3, 7) > 0.3
    attend[:, 0] = True

    expected, _ = mha(q, kv, kv, key_padding_mask=~attend, need_weights=False)
    actual = scaled_multihead_attention(mha, q, kv, kv, logit_scale=1.0, attend_mask=attend)

    torch.testing.assert_close(actual, expected)


def test_scale_multiplies_logits() -> None:
    # Scaling the logits by s equals scaling the query projection by s.
    torch.manual_seed(0)
    s = 2.5
    mha = nn.MultiheadAttention(D, H, batch_first=True).eval()
    sharpened = nn.MultiheadAttention(D, H, batch_first=True).eval()
    sharpened.load_state_dict(mha.state_dict())
    with torch.no_grad():
        sharpened.in_proj_weight[:D] *= s
        sharpened.in_proj_bias[:D] *= s
    x = torch.randn(2, 5, D)

    expected, _ = sharpened(x, x, x, need_weights=False)
    actual = scaled_multihead_attention(mha, x, x, x, logit_scale=s)

    torch.testing.assert_close(actual, expected)


def test_encoder_layer_unchanged_at_reference_size() -> None:
    torch.manual_seed(0)
    N = 9
    plain = AttentionLayer(hidden_dim=D, num_heads=H, ff_dim=32).eval()
    scaled = AttentionLayer(
        hidden_dim=D,
        num_heads=H,
        ff_dim=32,
        attention_scaling=AttentionScalingConfig(reference_size=N),
    ).eval()
    scaled.load_state_dict(plain.state_dict())
    x = torch.randn(2, N, D)

    torch.testing.assert_close(scaled(x), plain(x))
    assert not torch.allclose(scaled(torch.randn(2, 4 * N, D)), plain(torch.randn(2, 4 * N, D)))


def test_cached_attention_unchanged_at_reference_size() -> None:
    torch.manual_seed(0)
    B, R, N = 2, 3, 6
    plain = CachedMultiheadAttention(embed_dim=D, num_heads=H).eval()
    scaled = CachedMultiheadAttention(
        embed_dim=D, num_heads=H, attention_scaling=AttentionScalingConfig(reference_size=N)
    ).eval()
    scaled.load_state_dict(plain.state_dict())
    q, nodes = torch.randn(B, R, D), torch.randn(B, N, D)
    k, v = plain.project_kv(nodes)

    torch.testing.assert_close(scaled(q, k, v), plain(q, k, v))

    sharper = CachedMultiheadAttention(
        embed_dim=D, num_heads=H, attention_scaling=AttentionScalingConfig(reference_size=2)
    ).eval()
    sharper.load_state_dict(plain.state_dict())
    assert not torch.allclose(sharper(q, k, v), plain(q, k, v))


def test_wrapper_unchanged_at_reference_size() -> None:
    torch.manual_seed(0)
    N = 6
    plain = MultiheadAttentionWrapper(D, H).eval()
    scaled = MultiheadAttentionWrapper(
        D, H, attention_scaling=AttentionScalingConfig(reference_size=N)
    ).eval()
    scaled.load_state_dict(plain.state_dict())
    q, nodes = torch.randn(2, 3, 1, D), torch.randn(2, 3, N, D)
    attend = torch.ones(2, 3, N, dtype=torch.bool)
    attend[..., -1] = False

    torch.testing.assert_close(scaled(q, nodes, nodes, attend), plain(q, nodes, nodes, attend))


def test_scorer_scales_clipped_logits() -> None:
    torch.manual_seed(0)
    N = 8
    config = AttentionScalingConfig()
    plain = CompatibilityScorer(hidden_dim=D, use_projections=False)
    scaled = CompatibilityScorer(hidden_dim=D, use_projections=False, attention_scaling=config)
    context, nodes = torch.randn(2, 3, D), torch.randn(2, N, D)
    mask = torch.zeros(2, 1, N, dtype=torch.bool)

    expected = plain(context=context, node_embeddings=nodes, mask=mask, log_softmax=False)
    actual = scaled(context=context, node_embeddings=nodes, mask=mask, log_softmax=False)

    torch.testing.assert_close(actual, expected * config.factor(N))


@pytest.mark.parametrize(
    "model_config",
    [
        JointModelConfig(
            problem_name="cvrp",
            encoder=JointEncoderConfig(attention_scaling=AttentionScalingConfig()),
            decoder=AutoregressiveDecoderConfig(attention_scaling=AttentionScalingConfig()),
        ),
        SplitModelConfig(
            problem_name="cvrp",
            encoder=SplitEncoderConfig(
                num_solution_attention_layers=1,
                attention_scaling=AttentionScalingConfig(reference_size=21),
            ),
            decoder=SimpleOnePassDecoderConfig(
                attention_scaling=AttentionScalingConfig(reference_size=21)
            ),
        ),
        SplitModelConfig(
            problem_name="cvrp",
            decoder=SlotDecoderConfig(num_slots=5, attention_scaling=AttentionScalingConfig()),
        ),
    ],
    ids=["joint-ar", "split-simple", "split-slot"],
)
def test_models_roll_out_with_scaling(model_config) -> None:
    B, R, M = 3, 4, 5
    instance = CVRPSampler(CVRPDistribution.for_n(20))(B)
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    model = build_nds_model(model_config).eval()

    rollout = model.rollout(
        solution, rollouts_per_solution=R, deconstruction_size=M, rng=SeededGenerator(0)
    )

    assert rollout.actions.shape == (B, R, M)
    assert torch.isfinite(rollout.log_p).all()


def test_split_encoder_scales_every_attention_layer() -> None:
    config = AttentionScalingConfig(reference_size=101)
    model = build_nds_model(
        SplitModelConfig(
            problem_name="cvrp",
            encoder=SplitEncoderConfig(num_solution_attention_layers=1, attention_scaling=config),
        )
    )
    layers = [m for m in model.encoder.modules() if isinstance(m, AttentionLayer)]

    assert len(layers) == SplitEncoderConfig().num_instance_attention_layers + 1
    assert all(layer.attention_scaling == config for layer in layers)
