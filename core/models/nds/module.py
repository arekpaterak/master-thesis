from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim

from core.lns.operators.repair import RepairOperator
from core.lns.start_solution import with_warmup
from core.models.module import Module
from core.models.nds.apply_rollout import apply_rollout
from core.models.nds.baseline import build_baseline
from core.models.nds.config import TrainConfig
from core.models.nds.decoders.ar import AutoregressiveNDSDecoder
from core.models.nds.encoders.split import SplitNDSEncoder
from core.models.nds.factory import NDSModel
from core.models.nds.improvement_step import build_nds_improvement_step
from core.models.nds.policy_loss import build_policy_loss
from core.models.nds.reward import build_reward
from core.models.nds.training import (
    BatchTrainSummary,
    EpochTrainAccumulator,
    configure_training_step,
)
from core.models.nds.utils import RolloutMetricsRecorder
from core.models.nds.validation import (
    BatchValidationSummary,
    EpochValidationAccumulator,
    configure_validation_step,
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
            # SplitNDSEncoder has no single forward (its encode_instance/fuse would run eager
            # through an OptimizedModule proxy), so compile the static instance encoder it wraps;
            # the per-step fusion stays eager, matching the pre-refactor behaviour.
            if isinstance(self.model.encoder, SplitNDSEncoder):
                self.model.encoder.instance_encoder = torch.compile(
                    self.model.encoder.instance_encoder, **compile_kwargs
                )
            else:
                self.model.encoder = torch.compile(self.model.encoder, **compile_kwargs)
            # The autoregressive decoder has a Python decoding loop that does not compile
            # cleanly; only the one-pass decoders are compiled.
            if not isinstance(self.model.decoder, AutoregressiveNDSDecoder):
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
        apply_rollout_fn = (
            functools.partial(apply_rollout, stitch=True)
            if repair.requires_stitch
            else apply_rollout
        )

        training_improvement_step = build_nds_improvement_step(
            config.search, model=model, repair=repair
        )

        self._training_step_fn = configure_training_step(
            num_main_improvement_steps=config.trainer.num_main_improvement_steps,
            rollouts_per_solution=config.search.num_rollouts,
            num_initial_improvement_steps=config.search.num_improvement_steps,
            deconstruction_size=config.search.deconstruction_size,
            generate_start_solution=generate_training_start_solution,
            improvement_step=training_improvement_step,
            apply_rollout=apply_rollout_fn,
            compute_reward=build_reward(config.reward),
            policy_loss=build_policy_loss(config.loss),
            baseline=build_baseline(config.baseline),
            repair=repair,
            n_repairs=config.search.n_repairs,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=self._scaler,
            encode_once=getattr(config.model, "encode_once_in_training", False),
            improvement_step_progress=improvement_step_progress,
        )

        validation_rollout_recorder = RolloutMetricsRecorder()
        validation_improvement_step = build_nds_improvement_step(
            config.validation.search,
            model=model,
            repair=repair,
            recorder=validation_rollout_recorder,
        )

        self._validation_step_fn = configure_validation_step(
            generate_start_solution=generate_validation_start_solution,
            improvement_step=validation_improvement_step,
            search_params=config.validation.search,
            capture_snapshots_every_n_improvement_steps=config.validation.plot_every_n_improvement_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            step_progress=validation_improvement_step_progress,
            rollout_recorder=validation_rollout_recorder,
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
