from __future__ import annotations

from core.models.nds.config import ModelParams, SplitModelParams

from .decoders.config import (
    AutoregressiveDecoderConfig,
    DecoderConfig,
    ProximityCategoricalSelectorConfig,
    ProximityDecoderConfig,
    SimpleOnePassDecoderConfig,
    SlotDecoderConfig,
)
from .decoders.one_pass.base import OnePassDestroyDecoder
from .decoders.one_pass.proximity import ProximityCategoricalDecoder, ProximityDecoder
from .decoders.one_pass.simple import SimpleOnePassDecoder
from .decoders.one_pass.slot import SlotDecoder
from .models import NDSModel
from .models.ar import AutoregressiveNDSModel
from .models.lns_native import LNSNativeModel
from .models.one_pass import OnePassNDSModel


def build_model(model_params: ModelParams, problem_name: str) -> NDSModel:
    decoder_config = model_params.decoder
    shared_kwargs = dict(
        problem_name=problem_name,
        hidden_dim=model_params.hidden_dim,
        num_heads=model_params.num_heads,
        ff_dim=model_params.ff_dim,
        encoder_config=model_params.encoder,
        positional=model_params.positional_encoding,
    )

    if isinstance(model_params, SplitModelParams):
        if isinstance(decoder_config, AutoregressiveDecoderConfig):
            raise NotImplementedError(
                "The split-encoder (LNS-native) model does not support the autoregressive "
                "decoder yet; use a one-pass decoder."
            )
        return LNSNativeModel(
            **shared_kwargs, decoder=_build_one_pass_decoder(decoder_config, model_params)
        )

    joint_kwargs = dict(
        **shared_kwargs,
        with_neighbours_encoding=model_params.with_neighbours_encoding,
        with_route_encoding=model_params.with_route_encoding,
    )
    if isinstance(decoder_config, AutoregressiveDecoderConfig):
        return AutoregressiveNDSModel(
            **joint_kwargs,
            poly_dim=model_params.poly_dim,
            seed_dim=model_params.seed_dim,
            use_hidden_decoding_state=decoder_config.use_hidden_decoding_state,
        )
    return OnePassNDSModel(
        **joint_kwargs, decoder=_build_one_pass_decoder(decoder_config, model_params)
    )


def _build_one_pass_decoder(
    decoder_config: DecoderConfig, model_params: ModelParams
) -> OnePassDestroyDecoder:
    shared = dict(
        hidden_dim=model_params.hidden_dim,
        seed_dim=model_params.seed_dim,
        poly_dim=model_params.poly_dim,
        num_heads=model_params.num_heads,
    )
    match decoder_config:
        case SimpleOnePassDecoderConfig():
            return SimpleOnePassDecoder(**shared)
        case SlotDecoderConfig():
            return SlotDecoder(**shared, k=decoder_config.k)
        case ProximityDecoderConfig():
            return ProximityDecoder(**shared, sigma=decoder_config.sigma)
        case ProximityCategoricalSelectorConfig():
            return ProximityCategoricalDecoder(
                **shared,
                sigma_init=decoder_config.sigma_init,
                anisotropic=decoder_config.anisotropic,
            )
        case _:
            raise ValueError(f"Not a one-pass decoder config: {type(decoder_config).__name__}")
