from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import torch
from pydantic import BaseModel, Field, model_validator

from core.lns.config import SearchParams
from core.lns.operators.repair import RepairOperator
from core.methods.base import Method
from core.models.nds.config import JointModelConfig, ModelConfig
from core.models.nds.decoders.ar import AutoregressiveNDSDecoder
from core.models.nds.decoders.config import AutoregressiveDecoderConfig
from core.models.nds.encoders.config import JointEncoderConfig
from core.models.nds.factory import build_model
from core.models.nds.improvement_step import build_nds_improvement_step
from core.models.nds.pretrained.config import (
    ORIGINAL_NDS_ENCODING_ORDER,
    ORIGINAL_NDS_ROUTE_AGGREGATION,
    validate_encoder_matches_original_nds,
)
from core.problems import ProblemConfig


def _original_nds_model() -> JointModelConfig:
    """The architecture matching the original NDS pretrained checkpoint."""
    return JointModelConfig(
        encoder=JointEncoderConfig(
            encoding_order=ORIGINAL_NDS_ENCODING_ORDER,
            route_aggregation=ORIGINAL_NDS_ROUTE_AGGREGATION,
        ),
        decoder=AutoregressiveDecoderConfig(),
    )


class NDSPretrainedConfig(BaseModel):
    """Config for evaluating the original NDS pretrained checkpoint as a baseline."""

    type: Literal["nds-pretrained"] = "nds-pretrained"
    checkpoint_path: Path = Field(description="Path to the original NDS .pt checkpoint.")
    model: ModelConfig = Field(
        default_factory=_original_nds_model,
        description="Architecture to instantiate before loading the checkpoint.",
    )
    search: SearchParams = Field(default_factory=SearchParams)

    @model_validator(mode="after")
    def _validate_encoder_for_original_pretrained(self) -> Self:
        validate_encoder_matches_original_nds(self.model.encoder)
        return self


class NDSPretrainedMethod:
    @classmethod
    def build(
        cls,
        config: NDSPretrainedConfig,
        *,
        problem: ProblemConfig,
        generate_start_solution,
        device: torch.device,
    ) -> Method:
        from core.models.nds.pretrained.load import (
            load_original_pretrained_decoder_weights,
            load_original_pretrained_encoder_weights,
        )

        model = build_model(config.model, problem.name).to(device)
        if not isinstance(getattr(model, "decoder", None), AutoregressiveNDSDecoder):
            raise ValueError(
                "Expected the original NDS checkpoint to be used with an autoregressive decoder."
            )

        load_original_pretrained_encoder_weights(model.encoder, config.checkpoint_path)
        load_original_pretrained_decoder_weights(model.decoder, config.checkpoint_path)

        model.eval()

        repair = RepairOperator.create(
            problem.name,
            config.search.repair_type,
            deconstruction_size=config.search.deconstruction_size,
        )
        improvement_step = build_nds_improvement_step(config.search, model=model, repair=repair)

        return Method(
            search=config.search,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
        )
