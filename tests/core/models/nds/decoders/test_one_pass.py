import pytest
import torch

from core.models.nds.decoders.config import SimpleOnePassDecoderConfig, SlotDecoderConfig
from core.models.nds.decoders.factory import build_decoder
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
    query_context=True,
    skip_mha=False,
    attend_context=False,
    num_context_nodes=0,
):
    return SimpleOnePassDecoder(
        hidden_dim=HIDDEN_DIM,
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
        num_heads=NUM_HEADS,
        with_poly=with_poly,
        query_pooling=query_pooling,
        query_learned=query_learned,
        query_context=query_context,
        skip_mha=skip_mha,
        attend_to_context=attend_context,
        num_context_nodes=num_context_nodes,
    ), {}


def _slot(
    slot_self_attention="post",
    with_poly=True,
    node_pooling="none",
    node_context=True,
    attend_context=False,
    num_context_nodes=0,
):
    return SlotDecoder(
        hidden_dim=HIDDEN_DIM,
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
        num_heads=NUM_HEADS,
        num_slots=M,
        slot_self_attention=slot_self_attention,
        with_poly=with_poly,
        node_pooling=node_pooling,
        node_context=node_context,
        attend_to_context=attend_context,
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
    decoder = build_decoder(
        SimpleOnePassDecoderConfig(
            query_pooling=query_pooling,
            query_learned=query_learned,
            seed_dim=SEED_DIM,
            poly_dim=POLY_DIM,
        ),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.query.pooling == query_pooling
    assert (decoder.query.learned is not None) == query_learned


def test_simple_skip_mha_produces_logits():
    decoder, _ = _simple(skip_mha=True)
    assert decoder.MHA is None
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, N)


def test_simple_skip_mha_is_wired_to_factory():
    decoder = build_decoder(
        SimpleOnePassDecoderConfig(skip_mha=True, seed_dim=SEED_DIM, poly_dim=POLY_DIM),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.skip_mha is True
    assert decoder.MHA is None


def test_simple_context_query_produces_logits():
    decoder, _ = _simple(num_context_nodes=1)
    assert decoder.query.with_context is True
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, N)


def test_simple_context_off_without_context_nodes():
    decoder, _ = _simple(num_context_nodes=0)
    assert decoder.query.with_context is False


def test_simple_query_context_toggle_disables_context():
    decoder, _ = _simple(num_context_nodes=1, query_context=False)
    assert decoder.query.with_context is False


@pytest.mark.parametrize("num_context_nodes, expected", [(1, True), (0, False)])
def test_simple_context_is_wired_to_factory(num_context_nodes, expected):
    decoder = build_decoder(
        SimpleOnePassDecoderConfig(seed_dim=SEED_DIM, poly_dim=POLY_DIM),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_context_nodes=num_context_nodes,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.num_context_nodes == num_context_nodes
    assert decoder.query.with_context is expected


def test_simple_attend_context_defaults_off():
    decoder, _ = _simple(num_context_nodes=1)
    assert decoder.attend_to_context is False


def test_simple_attend_context_produces_logits():
    decoder, _ = _simple(num_context_nodes=1, attend_context=True)
    assert decoder.attend_to_context is True
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
    decoder = build_decoder(
        SimpleOnePassDecoderConfig(
            attend_to_context=attend_context, seed_dim=SEED_DIM, poly_dim=POLY_DIM
        ),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_context_nodes=1,
    )

    assert isinstance(decoder, SimpleOnePassDecoder)
    assert decoder.attend_to_context is attend_context


@pytest.mark.parametrize("slot_self_attention", ["none", "pre", "post"])
def test_slot_self_attention_config_is_wired_to_factory(slot_self_attention):
    decoder = build_decoder(
        SlotDecoderConfig(
            num_slots=M,
            slot_self_attention=slot_self_attention,
            seed_dim=SEED_DIM,
            poly_dim=POLY_DIM,
        ),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.slot_self_attention == slot_self_attention


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
    decoder = build_decoder(
        SlotDecoderConfig(
            num_slots=M,
            node_pooling=node_pooling,
            seed_dim=SEED_DIM,
            poly_dim=POLY_DIM,
        ),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.node_pooling == node_pooling


def test_slot_context_grounds_without_pooling():
    # With no pooling but context nodes present, the depot still reaches the slot query.
    decoder, _ = _slot(node_pooling="none", num_context_nodes=1)
    assert decoder.node_query is not None
    assert decoder.node_query.with_context is True
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, M, N)


def test_slot_no_grounding_without_pooling_or_context():
    decoder, _ = _slot(node_pooling="none", num_context_nodes=0)
    assert decoder.node_query is None


def test_slot_context_is_wired_to_factory():
    decoder = build_decoder(
        SlotDecoderConfig(num_slots=M, node_pooling="none", seed_dim=SEED_DIM, poly_dim=POLY_DIM),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_context_nodes=1,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.node_query is not None
    assert decoder.node_query.with_context is True


def test_slot_attend_context_produces_logits():
    decoder, _ = _slot(num_context_nodes=1, attend_context=True)
    assert decoder.attend_to_context is True
    h_nodes = torch.randn(B, N, HIDDEN_DIM)
    seed = torch.randn(B, R, SEED_DIM)
    action_mask = torch.zeros(B, N, dtype=torch.bool)
    action_mask[..., 0] = True  # the context node (depot) is unselectable

    logits = decoder.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)

    assert logits.shape == (B, R, M, N)
    assert torch.isinf(logits[..., 0]).all()


@pytest.mark.parametrize("attend_context", [True, False])
def test_slot_attend_context_is_wired_to_factory(attend_context):
    decoder = build_decoder(
        SlotDecoderConfig(
            num_slots=M, attend_to_context=attend_context, seed_dim=SEED_DIM, poly_dim=POLY_DIM
        ),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_context_nodes=1,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.attend_to_context is attend_context


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
