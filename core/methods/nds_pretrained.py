from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel, Field

from core.lns.acceptance_criterion import build_criterion
from core.lns.config import SearchParams
from core.lns.operators.repair import RepairOperator
from core.methods.base import Method
from core.models.nds.config import ModelParams, TrainConfig
from core.models.nds.factory import build_model
from core.models.nds.improvement_step import resolve_improvement_step
from core.models.nds.models.ar import AutoregressiveNDSModel
from core.problems.config import ProblemParams


class NDSPretrainedConfig(BaseModel):
    """Config for evaluating the original NDS pretrained checkpoint as a baseline."""

    checkpoint_path: Path = Field(description="Path to the original NDS .pt checkpoint.")
    load_decoder: bool = Field(default=False, description="Also load decoder weights.")
    model: ModelParams = Field(
        default_factory=ModelParams,
        description="Architecture to instantiate before loading the checkpoint.",
    )
    search: SearchParams = Field(
        default_factory=SearchParams, description="Per-method search knobs."
    )
    name_override: str | None = Field(
        default=None, description="Override the auto-derived method name."
    )


class NDSPretrainedMethod:
    @classmethod
    def build(
        cls,
        config: NDSPretrainedConfig,
        *,
        problem: ProblemParams,
        repair: RepairOperator,
        generate_start_solution,
        device: torch.device,
    ) -> Method:
        from core.models.nds.pretrained import (
            load_original_pretrained_decoder_weights,
            load_original_pretrained_encoder_weights,
        )

        # TODO: Adjust build_model to take a more minimal config object instead of TrainConfig.
        train_config = TrainConfig(
            problem=problem,
            model=config.model,
            search=config.search,
        )
        model = build_model(train_config).to(device)
        if not isinstance(model, AutoregressiveNDSModel):
            raise ValueError(
                "Expected the original NDS checkpoint to be used with an autoregressive model."
            )

        load_original_pretrained_encoder_weights(model.encoder, config.checkpoint_path)
        if config.load_decoder:
            load_original_pretrained_decoder_weights(model.decoder, config.checkpoint_path)

        model.eval()

        step_cls = resolve_improvement_step(config.search.improvement_step_type)
        improvement_step = step_cls(
            model=model,
            repair=repair,
            num_rollouts=config.search.num_rollouts,
            deconstruction_size=config.search.deconstruction_size,
            n_repairs=config.search.n_repairs,
        )
        criterion = build_criterion(
            config.search.acceptance, n_steps=config.search.num_improvement_steps
        )

        if config.name_override:
            name = config.name_override
        else:
            decoder_type = config.model.decoder.type
            assert decoder_type in ["ar"], "Expected the original decoder"
            hidden_tag = (
                ""
                if (decoder_type != "ar" or config.model.decoder.use_hidden_decoding_state)
                else "-nohidden"
            )
            decoder_tag = "+decoder" if config.load_decoder else ""
            name = f"nds-pretrained-{decoder_type}{hidden_tag}{decoder_tag}"

        return Method(
            name=name,
            search=config.search,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
            criterion=criterion,
        )
