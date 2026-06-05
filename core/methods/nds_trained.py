from __future__ import annotations

from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, Field

from core.lns.config import SearchParams
from core.methods.base import Method
from core.models.nds.config import TrainConfig
from core.models.nds.factory import build_model
from core.models.nds.improvement_step import resolve_improvement_step
from core.problems.setup import resolve_repair


class NDSTrainedConfig(BaseModel):
    run_dir: Path = Field(description="Path to the training run directory.")
    search: SearchParams = Field(default_factory=SearchParams)


def _find_latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoint found under {run_dir / 'checkpoints'}.")
    return checkpoints[-1]


class NDSTrainedMethod:
    @classmethod
    def build(
        cls,
        config: NDSTrainedConfig,
        *,
        generate_start_solution,
        device: torch.device,
    ) -> Method:
        train_config = TrainConfig.model_validate(
            yaml.safe_load((config.run_dir / "config.yml").read_text())
        )
        model = build_model(train_config).to(device)
        ckpt = torch.load(
            _find_latest_checkpoint(config.run_dir), map_location=device, weights_only=True
        )
        model.load_state_dict(ckpt["model"])
        model.eval()

        repair = resolve_repair(
            train_config.problem.name,
            config.search.repair_type,
            deconstruction_size=config.search.deconstruction_size,
        )
        step_cls = resolve_improvement_step(config.search.improvement_step_type)
        improvement_step = step_cls(
            model=model,
            repair=repair,
            num_rollouts=config.search.num_rollouts,
            deconstruction_size=config.search.deconstruction_size,
            n_repairs=config.search.n_repairs,
        )
        name = (
            f"nds-{train_config.model.decoder.type}"
            f"-d{config.search.deconstruction_size}-r{config.search.n_repairs}"
            f"-{config.search.improvement_step_type}"
        )

        return Method(
            name=name,
            search=config.search,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
        )
