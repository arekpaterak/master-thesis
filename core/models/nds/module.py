from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim

from core.lns.acceptance_criterion import build_criterion
from core.lns.config import SearchParams
from core.lns.operators.repair import RepairOperator
from core.lns.start_solution import with_warmup
from core.models.module import Module
from core.models.nds.apply_rollout import (
    apply_rollout,
    only_expanding_apply_rollout,
)
from core.models.nds.config import TrainConfig
from core.models.nds.evaluation import (
    BatchValidationSummary,
    EpochValidationAccumulator,
    configure_validation_step,
)
from core.models.nds.factory import NDSModel
from core.models.nds.improvement_step import resolve_improvement_step
from core.models.nds.models.lns_native import LNSNativeModel
from core.models.nds.models.one_pass import OnePassNDSModel
from core.models.nds.training import (
    BatchTrainSummary,
    EpochTrainAccumulator,
    configure_training_step,
)
from core.problems.instance import BaseProblemInstance
from core.problems.solve import generate_initial_solution


class NDSModule(Module):
    train_accumulator_cls = EpochTrainAccumulator
    validation_accumulator_cls = EpochValidationAccumulator

    def __init__(
        self,
        *,
        config: TrainConfig,
        model: NDSModel,
        improvement_step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        validation_improvement_step_progress: Callable[
            [Iterable[int]], Iterable[int]
        ] = lambda it: it,
    ) -> None:
        self.model = model

        if config.compile.enabled:
            compile_kwargs = dict(mode=config.compile.mode, fullgraph=config.compile.fullgraph)
            self.model.encoder = torch.compile(self.model.encoder, **compile_kwargs)
            if isinstance(self.model, (OnePassNDSModel, LNSNativeModel)):
                self.model.decoder = torch.compile(self.model.decoder, **compile_kwargs)

        amp_dtype = torch.float16 if config.amp.dtype == "float16" else torch.bfloat16
        amp_enabled = config.amp.enabled
        model_device = next(model.parameters()).device
        # GradScaler is only needed for float16. bfloat16 has fp32-equivalent range.
        self._scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler(device=model_device.type)
            if amp_enabled and model_device.type == "cuda" and amp_dtype == torch.float16
            else None
        )

        generate_training_start_solution = with_warmup(
            generate_initial_solution, problem_name=config.problem.name, search=config.search
        )
        generate_validation_start_solution = with_warmup(
            generate_initial_solution,
            problem_name=config.problem.name,
            search=config.validation.search,
        )
        repair = RepairOperator.create(
            config.problem.name,
            config.search.repair_type,
            deconstruction_size=max(
                config.search.deconstruction_size,
                config.validation.search.deconstruction_size,
            ),
        )
        if config.search.repair_type == "noop":
            apply_rollout_fn = only_expanding_apply_rollout
        elif repair.requires_stitch:
            apply_rollout_fn = functools.partial(apply_rollout, stitch=True)
        else:
            apply_rollout_fn = apply_rollout

        training_improvement_step = self._build_improvement_step(model, repair, config.search)

        self._training_step_fn = configure_training_step(
            num_main_improvement_steps=config.trainer.num_main_improvement_steps,
            rollouts_per_solution=config.search.num_rollouts,
            num_initial_improvement_steps=config.search.num_improvement_steps,
            deconstruction_size=config.search.deconstruction_size,
            generate_start_solution=generate_training_start_solution,
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
            generate_start_solution=generate_validation_start_solution,
            improvement_step=validation_improvement_step,
            criterion=validation_criterion,
            num_improvement_steps=config.validation.search.num_improvement_steps,
            n_augmentations=config.validation.search.n_augmentations,
            augmented_threshold_factor=config.validation.search.augmented_threshold_factor,
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
    def _build_improvement_step(
        model: NDSModel,
        repair: RepairOperator,
        search: SearchParams,
    ):
        step_cls = resolve_improvement_step(search.improvement_step_type)
        return step_cls(
            model=model,
            repair=repair,
            num_rollouts=search.num_rollouts,
            deconstruction_size=search.deconstruction_size,
            n_repairs=search.n_repairs,
            first_repair_uses_rollout_order=search.first_repair_uses_rollout_order,
            insert_in_new_routes_only=search.insert_in_new_routes_only,
        )
