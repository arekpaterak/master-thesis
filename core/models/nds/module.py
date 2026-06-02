from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim

from core.lns.acceptance_criterion import build_criterion
from core.lns.config import SearchParams
from core.lns.operators.repair import RepairOperator
from core.models.module import Module
from core.models.nds.config import TrainConfig
from core.models.nds.evaluation import (
    BatchValidationSummary,
    EpochValidationAccumulator,
    configure_validation_step,
)
from core.models.nds.factory import NDSModel
from core.models.nds.functions import (
    apply_rollout as default_apply_rollout,
)
from core.models.nds.functions import (
    only_expanding_apply_rollout,
)
from core.models.nds.improvement_step import resolve_improvement_step
from core.models.nds.training import (
    BatchTrainSummary,
    EpochTrainAccumulator,
    configure_training_step,
)
from core.problems.instance import BaseProblemInstance
from core.problems.setup import RepairType, resolve_repair, resolve_start_solution


class NDSModule(Module):
    train_accumulator_cls = EpochTrainAccumulator
    validation_accumulator_cls = EpochValidationAccumulator

    def __init__(
        self,
        *,
        config: TrainConfig,
        model: NDSModel,
        repair_type: RepairType = "greedy_reconnect",
        improvement_step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        validation_improvement_step_progress: Callable[
            [Iterable[int]], Iterable[int]
        ] = lambda it: it,
    ) -> None:
        self.model = model

        amp_dtype = torch.float16 if config.amp.dtype == "float16" else torch.bfloat16
        amp_enabled = config.amp.enabled
        model_device = next(model.parameters()).device
        # GradScaler is only needed for float16. bfloat16 has fp32-equivalent range.
        self._scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler(device=model_device.type)
            if amp_enabled and model_device.type == "cuda" and amp_dtype == torch.float16
            else None
        )

        generate_start_solution = resolve_start_solution(config.problem.name)
        repair = resolve_repair(
            config.problem.name,
            repair_type,
            deconstruction_size=max(
                config.search.deconstruction_size,
                config.validation.search.deconstruction_size,
            ),
        )
        apply_rollout_fn = (
            only_expanding_apply_rollout if repair_type == "noop" else default_apply_rollout
        )

        training_improvement_step = self._build_improvement_step(model, repair, config.search)

        self._training_step_fn = configure_training_step(
            num_main_improvement_steps=config.trainer.num_main_improvement_steps,
            rollouts_per_solution=config.search.num_rollouts,
            num_initial_improvement_steps=config.search.num_improvement_steps,
            deconstruction_size=config.search.deconstruction_size,
            generate_start_solution=generate_start_solution,
            improvement_step=training_improvement_step,
            apply_rollout=apply_rollout_fn,
            repair=repair,
            n_repairs=config.search.n_repairs,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=self._scaler,
            improvement_step_progress=improvement_step_progress,
        )

        validation_improvement_step = self._build_improvement_step(
            model, repair, config.validation.search
        )
        validation_criterion = build_criterion(
            config.validation.search.acceptance,
            n_steps=config.validation.search.num_improvement_steps,
        )

        self._validation_step_fn = configure_validation_step(
            generate_start_solution=generate_start_solution,
            improvement_step=validation_improvement_step,
            criterion=validation_criterion,
            num_improvement_steps=config.validation.search.num_improvement_steps,
            capture_snapshots_every_n_improvement_steps=config.validation.plot_every_n_improvement_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            step_progress=validation_improvement_step_progress,
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
        return self._training_step_fn(self.model, optimizer, instance)

    def validation_step(
        self,
        instance: BaseProblemInstance,
        epoch: int,
    ) -> BatchValidationSummary:
        return self._validation_step_fn(self.model, instance, epoch)

    @staticmethod
    def _build_improvement_step(model: NDSModel, repair: RepairOperator, search: SearchParams):
        step_cls = resolve_improvement_step(search.improvement_step_type)
        return step_cls(
            model=model,
            repair=repair,
            num_rollouts=search.num_rollouts,
            deconstruction_size=search.deconstruction_size,
            n_repairs=search.n_repairs,
        )
