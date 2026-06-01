from __future__ import annotations

from collections.abc import Callable, Iterator

import torch.nn as nn
import torch.optim as optim

from core.lns.improvement_step import ImprovementStep, NoopImprovementStep
from core.lns.operators.repair import RepairOperator
from core.models.module import Module
from core.models.nds.config import TrainConfig
from core.models.nds.evaluation import (
    BatchValidationSummary,
    EpochValidationAccumulator,
    configure_validation_step,
)
from core.models.nds.factory import NDSModel
from core.models.nds.functions import apply_rollout as default_apply_rollout
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
        apply_rollout: Callable[[BaseSolution, Rollout], BaseSolution] | None = None,
        repair: RepairOperator,
        improvement_step: ImprovementStep | None,
        on_improvement_step: Callable[[int, float], None] | None = None,
        on_validation_improvement_step: Callable[[int, float], None] | None = None,
    ) -> None:
        self.model = model
        apply_rollout_fn = default_apply_rollout if apply_rollout is None else apply_rollout
        self._training_step_fn = configure_training_step(
            num_main_improvement_steps=config.trainer.num_main_improvement_steps,
            rollouts_per_solution=config.search.num_rollouts,
            num_initial_improvement_steps=config.search.num_improvement_steps,
            deconstruction_size=config.search.deconstruction_size,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step or NoopImprovementStep(),
            apply_rollout=apply_rollout_fn,
            repair=repair,
            on_improvement_step=on_improvement_step,
        )
        self._validation_step_fn = configure_validation_step(
            generate_start_solution=generate_start_solution,
            apply_rollout_fn=apply_rollout_fn,
            repair=repair,
            deconstruction_size=config.validation.search.deconstruction_size,
            num_improvement_steps=config.validation.search.num_improvement_steps,
            capture_snapshots_every_n_improvement_steps=config.validation.plot_every_n_improvement_steps,
            rollouts_per_step=config.validation.search.num_rollouts,
            n_repairs=config.validation.search.n_repairs,
            temperature_start=config.validation.search.temperature_start,
            temperature_min=config.validation.search.temperature_min,
            improvement_step_type=config.validation.search.improvement_step_type,
            on_improvement_step=on_validation_improvement_step,
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
