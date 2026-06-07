from __future__ import annotations

from core.models.nds.config import ModelParams

from .decoders.config import (
    AutoregressiveDecoderConfig,
    ProximityDecoderConfig,
    SimpleOnePassDecoderConfig,
    SlotDecoderConfig,
)
from .decoders.one_pass.proximity import ProximityDecoder
from .decoders.one_pass.simple import SimpleOnePassDecoder
from .decoders.one_pass.slot import SlotDecoder
from .models import NDSModel
from .models.ar import AutoregressiveNDSModel
from .models.one_pass import OnePassNDSModel


def build_model(model: ModelParams, problem_name: str) -> NDSModel:
    decoder_config = model.decoder
    shared_kwargs = dict(
        problem_name=problem_name,
        hidden_dim=model.hidden_dim,
        num_heads=model.num_heads,
        ff_dim=model.ff_dim,
        with_neighbours_encoding=model.with_neighbours_encoding,
        with_route_encoding=model.with_route_encoding,
    )

    match decoder_config:
        case AutoregressiveDecoderConfig():
            return AutoregressiveNDSModel(
                **shared_kwargs,
                poly_dim=model.poly_dim,
                seed_dim=model.seed_dim,
                use_hidden_decoding_state=decoder_config.use_hidden_decoding_state,
            )
        case SimpleOnePassDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=SimpleOnePassDecoder(
                    hidden_dim=model.hidden_dim,
                    seed_dim=model.seed_dim,
                    poly_dim=model.poly_dim,
                    num_heads=model.num_heads,
                ),
            )
        case SlotDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=SlotDecoder(
                    hidden_dim=model.hidden_dim,
                    seed_dim=model.seed_dim,
                    poly_dim=model.poly_dim,
                    num_heads=model.num_heads,
                    k=decoder_config.k,
                ),
            )
        case ProximityDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=ProximityDecoder(
                    hidden_dim=model.hidden_dim,
                    seed_dim=model.seed_dim,
                    poly_dim=model.poly_dim,
                    num_heads=model.num_heads,
                    sigma=decoder_config.sigma,
                ),
            )
