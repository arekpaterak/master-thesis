import pytest
import torch

from core.models.layers.cross_attention import CrossAttentionConfig
from core.models.layers.decoder_query import DecoderQuery, DecoderQueryConfig
from core.models.nds.decoders.config import SimpleOnePassDecoderConfig, SlotDecoderConfig
from core.models.nds.decoders.one_pass.simple import SimpleOnePassDecoder
from core.models.nds.decoders.one_pass.slot import SlotDecoder

B, R, N, M = 2, 4, 10, 3
HIDDEN_DIM = 16
SEED_DIM = 8
POLY_DIM = 32
NUM_HEADS = 2


def _simple(
    with_poly=True,
    query_pooling="mean",
    query_learned=False,
    context_handling="separate",
    skip_mha=False,
    attend_context=False,
    cross_attn_residual=False,
    num_context_nodes=0,
):
    return SimpleOnePassDecoder(
        hidden_dim=HIDDEN_DIM,
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
        num_heads=NUM_HEADS,
        with_poly=with_poly,
        query=DecoderQueryConfig(
            pooling=query_pooling,
            learned=query_learned,
            context_handling=context_handling,
        ),
        cross_attention=CrossAttentionConfig(
            skip=skip_mha, attend_to_context=attend_context, residual=cross_attn_residual
        ),
        num_context_nodes=num_context_nodes,
    ), {}


def _slot(
    slot_self_attention="post",
    with_poly=True,
    node_pooling="none",
    context_handling="merge",
    attend_context=False,
    cross_attn_residual=False,
    skip_cross_attn=False,
    num_context_nodes=0,
):
    return SlotDecoder(
        hidden_dim=HIDDEN_DIM,
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
        num_heads=NUM_HEADS,
        num_slots=M,
        slot_self_attention=slot_self_attention,
        cross_attention=CrossAttentionConfig(
            skip=skip_cross_attn, attend_to_context=attend_context, residual=cross_attn_residual
        ),
        with_poly=with_poly,
        query=DecoderQueryConfig(pooling=node_pooling, context_handling=context_handling),
        num_context_nodes=num_context_nodes,
    ), {}


# (query_pooling, query_learned): the two orthogonal query components, including the
# learned-only and learned+pooled cells the two-axis contract now allows.
QUERY_COMPONENTS = [
    pytest.param("mean", False, id="mean"),
    pytest.param("max", False, id="max"),
    pytest.param("meanmax", False, id="meanmax"),
    pytest.param("none", True, id="learned"),
    pytest.param("mean", True, id="mean+learned"),
]


DECODERS = [
    pytest.param(_simple, id="gumbel"),
    pytest.param(_slot, id="slot"),
]


@pytest.mark.parametrize("make_decoder", DECODERS)
def test_without_poly_greedy_is_seed_independent(make_decoder):
    decoder, extra_kwargs = make_decoder(with_poly=False)
    decoder.eval()
    assert decoder.poly_net is None
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    def greedy_actions(seed):
        return decoder.rollout(
            h_nodes=h_nodes,
            seed=seed,
            action_mask=action_mask,
            deconstruction_size=M,
            greedy=True,
            **extra_kwargs,
        ).actions

    # Without PolyNet the greedy decode ignores the seed entirely.
    seed_a = torch.randn(B, R, SEED_DIM)
    seed_b = torch.randn(B, R, SEED_DIM)
    assert torch.equal(greedy_actions(seed_a), greedy_actions(seed_b))


@pytest.mark.parametrize("make_decoder", DECODERS)
@pytest.mark.parametrize("greedy", [False, True])
def test_produces_M_actions(make_decoder, greedy):
    decoder, extra_kwargs = make_decoder()
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    rollout = decoder.rollout(
        h_nodes=h_nodes,
        seed=seed,
        action_mask=action_mask,
        deconstruction_size=M,
        greedy=greedy,
        **extra_kwargs,
    )

    assert rollout.actions.shape == (B, R, M)
    assert rollout.log_p.shape == (B, R)
    assert rollout.actions.ge(0).all()
    assert rollout.actions.lt(N).all()


@pytest.mark.parametrize("make_decoder", DECODERS)
def test_masked_nodes_not_selected(make_decoder):
    decoder, extra_kwargs = make_decoder()
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)
    action_mask[..., 0] = True  # forbid node 0

    rollout = decoder.rollout(
        h_nodes=h_nodes,
        seed=seed,
        action_mask=action_mask,
        deconstruction_size=M,
        greedy=False,
        **extra_kwargs,
    )

    assert (rollout.actions != 0).all()


@pytest.mark.parametrize("slot_self_attention", ["none", "pre", "post"])
def test_slot_self_attention_modes_produce_logits(slot_self_attention):
    decoder, _ = _slot(slot_self_attention)
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, M, N)


@pytest.mark.parametrize("query_pooling, query_learned", QUERY_COMPONENTS)
def test_simple_query_produces_logits(query_pooling, query_learned):
    decoder, _ = _simple(query_pooling=query_pooling, query_learned=query_learned)
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, N)


@pytest.mark.parametrize("query_pooling, query_learned", QUERY_COMPONENTS)
def test_simple_query_is_wired_to_factory(query_pooling, query_learned):
    decoder = SimpleOnePassDecoderConfig(
        query=DecoderQueryConfig(pooling=query_pooling, learned=query_learned),
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.query.pooling == query_pooling
    assert (decoder.query.learned is not None) == query_learned


def test_simple_skip_mha_produces_logits():
    decoder, _ = _simple(skip_mha=True)
    assert decoder.cross_attn.mha is None
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, N)


def test_simple_skip_mha_is_wired_to_factory():
    decoder = SimpleOnePassDecoderConfig(
        cross_attention=CrossAttentionConfig(skip=True), seed_dim=SEED_DIM, poly_dim=POLY_DIM
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.cross_attn.mha is None


def test_simple_cross_attn_residual_defaults_off():
    decoder, _ = _simple()
    assert decoder.cross_attn.residual is False


def test_simple_cross_attn_residual_produces_logits():
    decoder, _ = _simple(cross_attn_residual=True)
    assert decoder.cross_attn.residual is True
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, N)


def test_simple_cross_attn_residual_is_wired_to_factory():
    decoder = SimpleOnePassDecoderConfig(
        cross_attention=CrossAttentionConfig(residual=True),
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.cross_attn.residual is True


def test_simple_context_query_produces_logits():
    decoder, _ = _simple(num_context_nodes=1)
    assert decoder.query.context == "separate"
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, N)


def test_simple_context_merge_produces_logits():
    decoder, _ = _simple(num_context_nodes=1, context_handling="merge")
    assert decoder.query.context == "merge"
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, N)


def test_merge_query_equals_mean_over_all_nodes():
    query = DecoderQuery(hidden_dim=HIDDEN_DIM, pooling="mean", context="merge")
    selectable = torch.randn(B, N, HIDDEN_DIM)
    context = torch.randn(B, 1, HIDDEN_DIM)

    merged = query(selectable, context_embedded_nodes=context)  # (B, 1, D)
    expected = torch.cat([selectable, context], dim=-2).mean(dim=-2, keepdim=True)

    assert torch.allclose(merged, expected, atol=1e-6)


def test_simple_context_off_without_context_nodes():
    decoder, _ = _simple(num_context_nodes=0)
    assert decoder.query.context == "exclude"


def test_simple_context_handling_exclude_drops_context():
    decoder, _ = _simple(num_context_nodes=1, context_handling="exclude")
    assert decoder.query.context == "exclude"


@pytest.mark.parametrize("num_context_nodes, expected", [(1, "merge"), (0, "exclude")])
def test_simple_context_is_wired_to_factory(num_context_nodes, expected):
    decoder = SimpleOnePassDecoderConfig(seed_dim=SEED_DIM, poly_dim=POLY_DIM).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_context_nodes=num_context_nodes,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.num_context_nodes == num_context_nodes
    assert decoder.query.context == expected


def test_simple_attend_context_defaults_off():
    decoder, _ = _simple(num_context_nodes=1)
    assert decoder.cross_attn.attend_to_context is False


def test_simple_attend_context_produces_logits():
    decoder, _ = _simple(num_context_nodes=1, attend_context=True)
    assert decoder.cross_attn.attend_to_context is True
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)
    action_mask[..., 0] = True  # the context node (depot) is unselectable

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, N)
    # Attending to the context node must not make it selectable: it stays at -inf.
    assert torch.isinf(logits[..., 0]).all()


def test_simple_attend_context_does_not_mutate_action_mask():
    decoder, _ = _simple(num_context_nodes=1, attend_context=True)
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)
    action_mask[..., 0] = True
    original = action_mask.clone()

    decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert torch.equal(action_mask, original)


@pytest.mark.parametrize("attend_context", [True, False])
def test_simple_attend_context_is_wired_to_factory(attend_context):
    decoder = SimpleOnePassDecoderConfig(
        cross_attention=CrossAttentionConfig(attend_to_context=attend_context),
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_context_nodes=1,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.cross_attn.attend_to_context is attend_context


@pytest.mark.parametrize("slot_self_attention", ["none", "pre", "post"])
def test_slot_self_attention_config_is_wired_to_factory(slot_self_attention):
    decoder = SlotDecoderConfig(
        num_slots=M,
        slot_self_attention=slot_self_attention,
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.slot_self_attention == slot_self_attention


def test_slot_cross_attn_residual_defaults_off():
    decoder, _ = _slot()
    assert decoder.cross_attn.residual is False


def test_slot_cross_attn_residual_produces_logits():
    decoder, _ = _slot(cross_attn_residual=True)
    assert decoder.cross_attn.residual is True
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, M, N)


@pytest.mark.parametrize("cross_attn_residual", [True, False])
def test_slot_cross_attn_residual_is_wired_to_factory(cross_attn_residual):
    decoder = SlotDecoderConfig(
        num_slots=M,
        cross_attention=CrossAttentionConfig(residual=cross_attn_residual),
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.cross_attn.residual is cross_attn_residual


def test_slot_skip_cross_attn_produces_logits():
    decoder, _ = _slot(skip_cross_attn=True)
    assert decoder.cross_attn.mha is None
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, M, N)


def test_slot_skip_cross_attn_is_wired_to_factory():
    decoder = SlotDecoderConfig(
        num_slots=M,
        cross_attention=CrossAttentionConfig(skip=True),
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.cross_attn.mha is None


@pytest.mark.parametrize("node_pooling", ["none", "mean", "max", "meanmax"])
def test_slot_node_pooling_produces_logits(node_pooling):
    decoder, _ = _slot(node_pooling=node_pooling)
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, M, N)
    assert (decoder.node_query is None) == (node_pooling == "none")


@pytest.mark.parametrize("node_pooling", ["none", "mean", "max", "meanmax"])
def test_slot_node_pooling_is_wired_to_factory(node_pooling):
    decoder = SlotDecoderConfig(
        num_slots=M,
        query=DecoderQueryConfig(pooling=node_pooling),
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SlotDecoder)
    if node_pooling == "none":
        assert decoder.node_query is None
    else:
        assert decoder.node_query is not None
        assert decoder.node_query.pooling == node_pooling


def test_slot_context_does_not_ground_without_pooling():
    decoder, _ = _slot(node_pooling="none", num_context_nodes=1)
    assert decoder.node_query is None


def test_slot_no_grounding_without_pooling_or_context():
    decoder, _ = _slot(node_pooling="none", num_context_nodes=0)
    assert decoder.node_query is None


def test_slot_separate_context_grounds_without_pooling():
    decoder, _ = _slot(node_pooling="none", context_handling="separate", num_context_nodes=1)
    assert decoder.node_query is not None
    assert decoder.node_query.context == "separate"
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, M, N)


def test_slot_separate_context_is_wired_to_factory():
    decoder = SlotDecoderConfig(
        num_slots=M,
        query=DecoderQueryConfig(pooling="none", context_handling="separate"),
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_context_nodes=1,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.node_query is not None
    assert decoder.node_query.context == "separate"


def test_slot_attend_context_produces_logits():
    decoder, _ = _slot(num_context_nodes=1, attend_context=True)
    assert decoder.cross_attn.attend_to_context is True
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)
    action_mask[..., 0] = True  # the context node (depot) is unselectable

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, M, N)
    assert torch.isinf(logits[..., 0]).all()


@pytest.mark.parametrize("attend_context", [True, False])
def test_slot_attend_context_is_wired_to_factory(attend_context):
    decoder = SlotDecoderConfig(
        num_slots=M,
        cross_attention=CrossAttentionConfig(attend_to_context=attend_context),
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
    ).build(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_context_nodes=1,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.cross_attn.attend_to_context is attend_context


@pytest.mark.parametrize("greedy", [False, True])
def test_slot_actions_are_distinct(greedy):
    # Sequential masking must yield M distinct nodes per rollout, even with a forbidden node.
    decoder, _ = _slot()
    torch.manual_seed(0)
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)
    action_mask[..., 0] = True  # forbid node 0

    rollout = decoder.rollout(
        h_nodes=h_nodes,
        seed=seed,
        action_mask=action_mask,
        deconstruction_size=M,
        greedy=greedy,
    )

    for row in rollout.actions.reshape(-1, M):
        assert row.unique().numel() == M  # all distinct
    assert (rollout.actions != 0).all()  # forbidden node never selected
    assert torch.isfinite(rollout.log_p).all()
