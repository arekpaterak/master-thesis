from __future__ import annotations

from .ar import AutoregressiveNDSDecoder
from .base import DestroyDecoder
from .config import (
    AutoregressiveDecoderConfig,
    DecoderConfig,
    SimpleOnePassDecoderConfig,
    SlotDecoderConfig,
)
from .one_pass.simple import SimpleOnePassDecoder
from .one_pass.slot import SlotDecoder


def build_decoder(
    decoder_config: DecoderConfig,
    *,
    hidden_dim: int,
    num_heads: int,
) -> DestroyDecoder:
    """Instantiate the destroy decoder selected by ``decoder_config``."""
    shared = dict(
        hidden_dim=hidden_dim,
        seed_dim=decoder_config.seed_dim,
        poly_dim=decoder_config.poly_dim,
        num_heads=num_heads,
        with_poly=decoder_config.with_poly,
    )
    match decoder_config:
        case AutoregressiveDecoderConfig():
            return AutoregressiveNDSDecoder(
                **shared, use_hidden_decoding_state=decoder_config.use_hidden_decoding_state
            )
        case SimpleOnePassDecoderConfig():
            return SimpleOnePassDecoder(**shared)
        case SlotDecoderConfig():
            return SlotDecoder(
                **shared,
                num_slots=decoder_config.num_slots,
                slot_self_attention=decoder_config.slot_self_attention,
            )
        case _:
            raise ValueError(f"Unknown decoder config: {type(decoder_config).__name__}")
