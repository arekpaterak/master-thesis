from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, Field

from core.lns.config import SearchParams
from core.lns.operators.repair import RepairOperator
from core.methods.base import Method
from core.models.nds.config import TrainConfig
from core.models.nds.factory import build_model
from core.models.nds.improvement_step import build_nds_improvement_step


class NDSTrainedConfig(BaseModel):
    type: Literal["nds-trained"] = "nds-trained"
    run_dir: Path = Field(description="Path to the training run directory.")
    search: SearchParams = Field(default_factory=SearchParams)


def _find_latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoint found under {run_dir / 'checkpoints'}.")
    return checkpoints[-1]


def _load_train_config(run_dir: Path, checkpoint: dict) -> TrainConfig:
    embedded_config = checkpoint.get("config")
    if embedded_config is not None:
        return TrainConfig.model_validate(embedded_config)
    return TrainConfig.model_validate(yaml.safe_load((run_dir / "config.yml").read_text()))


class NDSTrainedMethod:
    @classmethod
    def build(
        cls,
        config: NDSTrainedConfig,
        *,
        generate_start_solution,
        device: torch.device,
    ) -> Method:
        ckpt = torch.load(
            _find_latest_checkpoint(config.run_dir), map_location=device, weights_only=True
        )
        train_config = _load_train_config(config.run_dir, ckpt)
        model = build_model(train_config.model, train_config.problem.name).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        repair = RepairOperator.create(
            train_config.problem.name,
            config.search.repair_type,
            deconstruction_size=config.search.deconstruction_size,
        )
        improvement_step = build_nds_improvement_step(config.search, model=model, repair=repair)

        return Method(
            search=config.search,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
        )
