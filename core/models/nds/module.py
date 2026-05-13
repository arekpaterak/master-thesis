from __future__ import annotations

from collections.abc import Callable, Iterator

import torch.nn as nn
import torch.optim as optim

from core.models.module import Module
from core.models.nds.config import TrainConfig
from core.models.nds.evaluation import (
    BatchValidationSummary,
    EpochValidationAccumulator,
    configure_validation_step,
)
from core.models.nds.functions import noop_improvement_step
from core.models.nds.model import NDSModel
from core.models.nds.training import (
    BatchTrainSummary,
    EpochTrainAccumulator,
    configure_training_step,
)
from core.models.rollout import Rollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class NDSModule(Module):
    train_accumulator_cls = EpochTrainAccumulator
    validation_accumulator_cls = EpochValidationAccumulator

    def __init__(
        self,
        *,
        config: TrainConfig,
        model: NDSModel,
        generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
        apply_rollout: Callable[[BaseSolution, Rollout], BaseSolution],
        repair: Callable[[BaseSolution, Rollout], BaseSolution],
        improvement_step: Callable[[BaseSolution, NDSModel], BaseSolution] | None = None,
        on_improvement_step: Callable[[int, float], None] | None = None,
    ) -> None:
        self.model = model
        self._training_step_fn = configure_training_step(
            num_main_improvement_steps=config.trainer.num_main_improvement_steps,
            rollouts_per_solution=config.trainer.rollouts_per_instance,
            num_initial_improvement_steps=config.trainer.num_initial_improvement_steps,
            deconstruction_size=config.trainer.deconstruction_size,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step or noop_improvement_step,
            apply_rollout=apply_rollout,
            repair=repair,
            on_improvement_step=on_improvement_step,
        )
        self._validation_step_fn = configure_validation_step(
            generate_start_solution=generate_start_solution,
            apply_rollout_fn=apply_rollout,
            repair=repair,
            deconstruction_size=config.trainer.deconstruction_size,
            num_improvement_steps=config.validation.num_improvement_steps,
            every_n_lns_steps=config.validation.plot_every_n_improvement_steps,
        )

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

    def training_step(
        self,
        *,
        optimizer: optim.Optimizer,
        instance: BaseProblemInstance,
        **_,
    ) -> BatchTrainSummary:
        return self._training_step_fn(model=self.model, optimizer=optimizer, instance=instance)

    def validation_step(
        self,
        instance: BaseProblemInstance,
        epoch: int,
    ) -> BatchValidationSummary:
        return self._validation_step_fn(self.model, instance, epoch)
