from __future__ import annotations

from core.models.nds.config import ModelConfig, SplitModelConfig
from core.problems import depot_offset

from .decoders.factory import build_decoder
from .models import NDSModel
from .models.joint import JointNDSModel
from .models.split import SplitNDSModel


def build_model(model_params: ModelConfig) -> NDSModel:
    decoder_config = model_params.decoder
    shared_kwargs = dict(
        problem_name=model_params.problem_name,
        hidden_dim=model_params.hidden_dim,
        num_heads=model_params.num_heads,
        ff_dim=model_params.ff_dim,
        encoder_config=model_params.encoder,
        positional=model_params.positional_encoding,
    )

    if isinstance(model_params, SplitModelConfig):
        return SplitNDSModel(
            **shared_kwargs,
            decoder=build_decoder(
                decoder_config,
                hidden_dim=model_params.hidden_dim,
                num_heads=model_params.num_heads,
                num_context_nodes=depot_offset(model_params.problem_name),
            ),
        )

    return JointNDSModel(
        **shared_kwargs,
        decoder=decoder_config,
    )
