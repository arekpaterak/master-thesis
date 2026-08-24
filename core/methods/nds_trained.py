from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, Field

from core.lns.config import SearchParams
from core.methods.base import Method
from core.models.nds.config import TrainConfig
from core.models.nds.factory import build_model
from core.models.nds.improvement_step import build_nds_improvement_step
from core.models.utils import checkpoint_epoch, find_checkpoint, list_checkpoints


class NDSTrainedConfig(BaseModel):
    type: Literal["nds-trained"] = "nds-trained"
    run_dir: Path = Field(description="Path to the training run directory.")
    checkpoint_epoch: int | None = Field(
        default=None,
        description="Epoch of the checkpoint to load (defaults to the latest checkpoint).",
    )
    search: SearchParams = Field(default_factory=SearchParams)

    def resolved_checkpoint_epoch(self) -> int | None:
        """Epoch of the checkpoint this config loads, or ``None`` if none exist yet."""
        if self.checkpoint_epoch is not None:
            return self.checkpoint_epoch
        checkpoints = list_checkpoints(self.run_dir)
        if not checkpoints:
            return None
        return checkpoint_epoch(checkpoints[-1])


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
            find_checkpoint(config.run_dir, config.checkpoint_epoch),
            map_location=device,
            weights_only=True,
        )
        train_config = _load_train_config(config.run_dir, ckpt)
        model = build_model(train_config.model).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        improvement_step = build_nds_improvement_step(config.search, model=model)

        return Method(
            search=config.search,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
        )
