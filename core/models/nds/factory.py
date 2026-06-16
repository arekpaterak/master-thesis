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


def build_model(model_params: ModelParams, problem_name: str) -> NDSModel:
    decoder_config = model_params.decoder
    shared_kwargs = dict(
        problem_name=problem_name,
        hidden_dim=model_params.hidden_dim,
        num_heads=model_params.num_heads,
        ff_dim=model_params.ff_dim,
        encoder_config=model_params.encoder,
        positional=model_params.positional,
        with_neighbours_encoding=model_params.with_neighbours_encoding,
        with_route_encoding=model_params.with_route_encoding,
    )

    match decoder_config:
        case AutoregressiveDecoderConfig():
            return AutoregressiveNDSModel(
                **shared_kwargs,
                poly_dim=model_params.poly_dim,
                seed_dim=model_params.seed_dim,
                use_hidden_decoding_state=decoder_config.use_hidden_decoding_state,
            )
        case SimpleOnePassDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=SimpleOnePassDecoder(
                    hidden_dim=model_params.hidden_dim,
                    seed_dim=model_params.seed_dim,
                    poly_dim=model_params.poly_dim,
                    num_heads=model_params.num_heads,
                ),
            )
        case SlotDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=SlotDecoder(
                    hidden_dim=model_params.hidden_dim,
                    seed_dim=model_params.seed_dim,
                    poly_dim=model_params.poly_dim,
                    num_heads=model_params.num_heads,
                    k=decoder_config.k,
                ),
            )
        case ProximityDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=ProximityDecoder(
                    hidden_dim=model_params.hidden_dim,
                    seed_dim=model_params.seed_dim,
                    poly_dim=model_params.poly_dim,
                    num_heads=model_params.num_heads,
                    sigma=decoder_config.sigma,
                ),
            )
