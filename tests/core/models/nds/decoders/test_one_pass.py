import pytest
import torch

from core.models.nds.decoders.config import SlotDecoderConfig
from core.models.nds.decoders.factory import build_decoder
from core.models.nds.decoders.one_pass.simple import SimpleOnePassDecoder
from core.models.nds.decoders.one_pass.slot import SlotDecoder

B, R, N, M = 2, 4, 10, 3
HIDDEN_DIM = 16
SEED_DIM = 8
POLY_DIM = 32
NUM_HEADS = 2


def _simple(with_poly=True):
    return SimpleOnePassDecoder(
        hidden_dim=HIDDEN_DIM,
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
        num_heads=NUM_HEADS,
        with_poly=with_poly,
    ), {}


def _slot(slot_self_attention="post", with_poly=True):
    return SlotDecoder(
        hidden_dim=HIDDEN_DIM,
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
        num_heads=NUM_HEADS,
        num_slots=M,
        slot_self_attention=slot_self_attention,
        with_poly=with_poly,
    ), {}


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


@pytest.mark.parametrize("slot_self_attention", ["none", "pre", "post"])
def test_slot_self_attention_config_is_wired_to_factory(slot_self_attention):
    decoder = build_decoder(
        SlotDecoderConfig(num_slots=M, slot_self_attention=slot_self_attention),
        hidden_dim=HIDDEN_DIM,
        seed_dim=SEED_DIM,
        poly_dim=POLY_DIM,
        num_heads=NUM_HEADS,
    )

    assert isinstance(decoder, SlotDecoder)
    assert decoder.slot_self_attention == slot_self_attention
