from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel, Field

from core.lns.config import SearchParams
from core.lns.operators.repair import RepairOperator
from core.methods.base import Method
from core.models.nds.config import ModelParams
from core.models.nds.factory import build_model
from core.models.nds.improvement_step import build_nds_improvement_step
from core.models.nds.models.ar import AutoregressiveNDSModel
from core.problems import ProblemConfig


class NDSPretrainedConfig(BaseModel):
    """Config for evaluating the original NDS pretrained checkpoint as a baseline."""

    checkpoint_path: Path = Field(description="Path to the original NDS .pt checkpoint.")
    load_decoder: bool = Field(default=False, description="Whether to also load decoder weights.")
    model: ModelParams = Field(
        default_factory=ModelParams,
        description="Architecture to instantiate before loading the checkpoint.",
    )
    search: SearchParams = Field(default_factory=SearchParams)


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
        from core.models.nds.pretrained import (
            load_original_pretrained_decoder_weights,
            load_original_pretrained_encoder_weights,
        )

        model = build_model(config.model, problem.name).to(device)
        if not isinstance(model, AutoregressiveNDSModel):
            raise ValueError(
                "Expected the original NDS checkpoint to be used with an autoregressive model."
            )

        load_original_pretrained_encoder_weights(model.encoder, config.checkpoint_path)
        if config.load_decoder:
            load_original_pretrained_decoder_weights(model.decoder, config.checkpoint_path)

        model.eval()

        repair = RepairOperator.create(
            problem.name,
            config.search.repair_type,
            deconstruction_size=config.search.deconstruction_size,
        )
        improvement_step = build_nds_improvement_step(config.search, model=model, repair=repair)
        decoder_type = config.model.decoder.type
        assert decoder_type in ["ar"], "Expected the original decoder"
        hidden_tag = (
            ""
            if (decoder_type != "ar" or config.model.decoder.use_hidden_decoding_state)
            else "-nohidden"
        )
        decoder_tag = "+decoder" if config.load_decoder else ""
        name = (
            f"nds-pretrained-{decoder_type}{hidden_tag}{decoder_tag}"
            f"-d{config.search.deconstruction_size}-r{config.search.n_repairs}"
            f"-{config.search.improvement_step_type}"
        )

        return Method(
            name=name,
            search=config.search,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
        )
