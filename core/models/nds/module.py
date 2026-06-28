from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim

from core.lns.operators.repair import RepairOperator
from core.lns.start_solution import with_warmup
from core.methods.base import Method
from core.models.module import Module
from core.models.nds.apply_rollout import apply_rollout
from core.models.nds.baseline import (
    GreedyRolloutBaseline,
    GreedyRolloutBaselineConfig,
    build_baseline,
)
from core.models.nds.config import TrainConfig
from core.models.nds.decoders.ar import AutoregressiveNDSDecoder
from core.models.nds.factory import NDSModel
from core.models.nds.improvement_step import build_nds_improvement_step
from core.models.nds.policy_loss import build_policy_loss
from core.models.nds.reward import build_reward
from core.models.nds.training import (
    BatchTrainSummary,
    EpochTrainAccumulator,
    EpochTrainSummary,
    configure_training_step,
)
from core.models.nds.validation import (
    BatchValidationSummary,
    EpochValidationAccumulator,
    configure_validation_step,
)
from core.models.statistical_test import OneSidedPairedTTestResult
from core.problems.instance import BaseProblemInstance
from core.problems.sampler import Sampler
from core.problems.solve import generate_initial_solution
from core.solve import solve


class NDSModule(Module):
    train_accumulator_cls = EpochTrainAccumulator
    validation_accumulator_cls = EpochValidationAccumulator

    def __init__(
        self,
        *,
        config: TrainConfig,
        model: NDSModel,
        baseline_eval_sampler: Sampler | None = None,
        on_baseline_sync: Callable[[OneSidedPairedTTestResult], None] | None = None,
        improvement_step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        validation_improvement_step_progress: Callable[
            [Iterable[int]], Iterable[int]
        ] = lambda it: it,
    ) -> None:
        self.model = model
        self._config = config
        self._baseline_eval_sampler = baseline_eval_sampler
        self._on_baseline_sync = on_baseline_sync

        if config.compile.enabled:
            compile_kwargs = dict(mode=config.compile.mode, fullgraph=config.compile.fullgraph)
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

        self._repair = repair
        self._generate_validation_start_solution = generate_validation_start_solution
        self.baseline = build_baseline(config.baseline)
        self.baseline.setup(model)
        if isinstance(config.baseline, GreedyRolloutBaselineConfig):
            self._eval_num_instances = config.baseline.num_instances
            self._eval_batch_size = config.baseline.batch_size
        else:
            self._eval_num_instances = 0
            self._eval_batch_size = 0

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
            baseline=self.baseline,
            repair=repair,
            n_repairs=config.search.n_repairs,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=self._scaler,
            improvement_step_progress=improvement_step_progress,
        )

        validation_improvement_step = build_nds_improvement_step(
            config.validation.search, model=model, repair=repair
        )

        self._validation_step_fn = configure_validation_step(
            generate_start_solution=generate_validation_start_solution,
            improvement_step=validation_improvement_step,
            search_params=config.validation.search,
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

    def on_epoch_train_end(self, epoch_summary: EpochTrainSummary) -> None:
        """Run the greedy-rollout baseline's periodic policy-vs-baseline t-test sync."""
        if not isinstance(self.baseline, GreedyRolloutBaseline):
            return
        if self._baseline_eval_sampler is None:
            return
        # Draw a fresh eval set each epoch; both policy and baseline are scored on these
        # same instances, so the paired t-test stays valid within the epoch.
        instances = self._baseline_eval_sampler(self._eval_num_instances)
        result = self.baseline.epoch_callback(
            policy=self.model,
            metric=lambda model: self._search_objectives(model, instances),
            epoch=epoch_summary.epoch,
        )
        if result is not None and self._on_baseline_sync is not None:
            self._on_baseline_sync(result)

    def _search_objectives(self, model: NDSModel, instances: BaseProblemInstance) -> torch.Tensor:
        """Run the validation search for ``model`` over ``instances`` and return the
        per-instance final objective.

        The metric for the greedy-baseline paired t-test. The same instances are scored
        for both the policy and the baseline, so the two objective vectors are paired.
        """
        search_params = self._config.validation.search
        device = next(model.parameters()).device
        model.eval()
        method = Method(
            search=search_params,
            generate_start_solution=self._generate_validation_start_solution,
            improvement_step=build_nds_improvement_step(
                search_params, model=model, repair=self._repair
            ),
        )

        objectives: list[torch.Tensor] = []
        num_instances = instances.batch_size[0]
        with torch.no_grad():
            for start in range(0, num_instances, self._eval_batch_size):
                chunk = instances[start : start + self._eval_batch_size].to(device)
                final = solve(method, chunk)
                objectives.append(final.objective().detach().cpu())
        return torch.cat(objectives)
