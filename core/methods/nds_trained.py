from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import Field

from core.lns.config import SearchParams
from core.methods.base import Method, MethodConfigBase
from core.models.nds.config import TrainConfig
from core.models.nds.improvement_step import build_nds_improvement_step
from core.models.utils import checkpoint_epoch, find_checkpoint, list_checkpoints
from core.problems import ProblemConfig, registry
from core.problems.registry import build_nds_model


class NDSTrainedConfig(MethodConfigBase):
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

    def build(self, *, problem: ProblemConfig, device: torch.device) -> Method:
        ckpt = torch.load(
            find_checkpoint(self.run_dir, self.checkpoint_epoch),
            map_location=device,
            weights_only=True,
        )
        train_config = _load_train_config(self.run_dir, ckpt)
        if train_config.model.problem_name != problem.name:
            raise ValueError(
                f"nds-trained model.problem_name '{train_config.model.problem_name}' does not "
                f"match the evaluation problem '{problem.name}'."
            )
        limit = train_config.model.decoder.max_deconstruction_size
        if limit is not None and self.search.deconstruction_size > limit:
            raise ValueError(
                f"The model in {self.run_dir} selects at most {limit} nodes, but the search "
                f"removes {self.search.deconstruction_size}."
            )
        model = build_nds_model(train_config.model).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        lns = registry.lns(problem.name)
        improvement_step = build_nds_improvement_step(self.search, model=model, lns=lns)
        return Method.from_lns(self.search, lns=lns, improvement_step=improvement_step)


def _load_train_config(run_dir: Path, checkpoint: dict) -> TrainConfig:
    embedded_config = checkpoint.get("config")
    if embedded_config is not None:
        return TrainConfig.model_validate(embedded_config)
    return TrainConfig.model_validate(yaml.safe_load((run_dir / "config.yml").read_text()))
