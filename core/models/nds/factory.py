from __future__ import annotations

from core.models.nds.config import (
    TrainConfig,
)
from core.models.nds.decoders.config import (
    AutoregressiveDecoderConfig,
    ProximityDecoderConfig,
    SimpleOnePassDecoderConfig,
    SlotDecoderConfig,
)
from core.models.nds.decoders.one_pass.proximity import ProximityDecoder
from core.models.nds.decoders.one_pass.simple import SimpleOnePassDecoder
from core.models.nds.decoders.one_pass.slot import SlotDecoder
from core.models.nds.models.ar import AutoregressiveNDSModel
from core.models.nds.models.one_pass import OnePassNDSModel

NDSModel = AutoregressiveNDSModel | OnePassNDSModel


def build_model(config: TrainConfig) -> NDSModel:
    decoder_config = config.model.decoder
    shared_kwargs = dict(
        problem_name=config.problem.name,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        ff_dim=config.model.ff_dim,
        with_neighbours_encoding=config.model.with_neighbours_encoding,
        with_tour_encoding=config.model.with_tour_encoding,
    )

    match decoder_config:
        case AutoregressiveDecoderConfig():
            return AutoregressiveNDSModel(
                **shared_kwargs,
                poly_dim=config.model.poly_dim,
                seed_dim=config.model.seed_dim,
                use_hidden_decoding_state=decoder_config.use_hidden_decoding_state,
            )
        case SimpleOnePassDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=SimpleOnePassDecoder(
                    hidden_dim=config.model.hidden_dim,
                    seed_dim=config.model.seed_dim,
                    poly_dim=config.model.poly_dim,
                    num_heads=config.model.num_heads,
                ),
            )
        case SlotDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=SlotDecoder(
                    hidden_dim=config.model.hidden_dim,
                    seed_dim=config.model.seed_dim,
                    poly_dim=config.model.poly_dim,
                    num_heads=config.model.num_heads,
                    k=config.trainer.deconstruction_size,
                ),
            )
        case ProximityDecoderConfig():
            return OnePassNDSModel(
                **shared_kwargs,
                decoder=ProximityDecoder(
                    hidden_dim=config.model.hidden_dim,
                    seed_dim=config.model.seed_dim,
                    poly_dim=config.model.poly_dim,
                    num_heads=config.model.num_heads,
                    sigma=decoder_config.sigma,
                ),
            )
