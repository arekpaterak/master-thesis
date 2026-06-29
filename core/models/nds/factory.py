from __future__ import annotations

from core.models.nds.config import ModelParams, SplitModelParams

from .decoders.factory import build_decoder
from .models import NDSModel
from .models.joint import JointNDSModel
from .models.split import SplitNDSModel


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
        return SplitNDSModel(
            **shared_kwargs,
            with_fusion_neighbours_encoding=model_params.with_fusion_neighbours_encoding,
            with_fusion_route_encoding=model_params.with_fusion_route_encoding,
            num_fusion_attention_layers=model_params.num_fusion_attention_layers,
            decoder=build_decoder(
                decoder_config,
                hidden_dim=model_params.hidden_dim,
                seed_dim=model_params.seed_dim,
                poly_dim=model_params.poly_dim,
                num_heads=model_params.num_heads,
            ),
        )

    return JointNDSModel(
        **shared_kwargs,
        with_neighbours_encoding=model_params.with_neighbours_encoding,
        with_route_encoding=model_params.with_route_encoding,
        poly_dim=model_params.poly_dim,
        seed_dim=model_params.seed_dim,
        decoder=decoder_config,
    )
