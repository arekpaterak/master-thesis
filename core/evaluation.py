from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Iterable, Iterator

import torch
from pydantic import BaseModel, computed_field

from core.eval.problem_setting import SyntheticProblemSetting
from core.methods.base import Method
from core.problems.dataset import DatasetSampler
from core.problems.instance import BaseProblemInstance
from core.problems.sampler import Sampler
from core.problems.solution import BaseSolution
from core.solve import solve


class EvaluationResult(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    step_labels: list[int]
    objective_per_step: torch.Tensor  # (n_steps, num_test_instances)
    time_per_step: list[float]  # total wall-clock (s) to reach each step, summed over batches
    total_time_s: float = 0.0

    @computed_field
    @property
    def objective_final(self) -> torch.Tensor:
        return self.objective_per_step[-1]


class BatchEvaluationResult(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    step_labels: list[int]
    objective_per_step: torch.Tensor  # (n_steps, batch_size)
    cumulative_time_per_step: list[float]  # wall-clock (s) since batch start, per step


StepCallback = Callable[[int, int, BaseSolution], None]  # (batch_idx, step, solution)
BatchCallback = Callable[[int, BatchEvaluationResult], None]  # (batch_idx, batch_result)


class Evaluator:
    def __init__(
        self,
        *,
        on_step: list[StepCallback] | None = None,
        on_batch_end: list[BatchCallback] | None = None,
        batch_progress: Callable[[Iterable], Iterable] = lambda it: it,
        step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        device: torch.device | None = None,
    ) -> None:
        self.on_step: list[StepCallback] = list(on_step) if on_step else []
        self.on_batch_end: list[BatchCallback] = list(on_batch_end) if on_batch_end else []
        self.batch_progress = batch_progress
        self.step_progress = step_progress
        self.device = device

    def evaluate(self, setting: SyntheticProblemSetting, method: Method) -> EvaluationResult:
        batches: list[BatchEvaluationResult] = []
        start = time.perf_counter()
        for batch_idx, instance in enumerate(
            self.batch_progress(list(self._iter_batches(setting)))
        ):
            batch_result = self._evaluate_batch(method, instance, batch_idx)
            for callback in self.on_batch_end:
                callback(batch_idx, batch_result)
            batches.append(batch_result)
        elapsed = time.perf_counter() - start
        return self._aggregate(batches, total_time_s=elapsed)

    def _iter_batches(self, setting: SyntheticProblemSetting) -> Iterator[BaseProblemInstance]:
        sampler = self._build_sampler(setting)
        num_batches = math.ceil(setting.num_test_instances / setting.batch_size)
        remaining = setting.num_test_instances
        for _ in range(num_batches):
            batch_size = min(setting.batch_size, remaining)
            remaining -= batch_size
            instance = sampler(batch_size)
            if self.device is not None:
                instance = instance.to(self.device)
            yield instance

    def _build_sampler(self, setting: SyntheticProblemSetting) -> Sampler:
        if setting.dataset_path is None:
            return Sampler.from_config(setting.problem, seed=setting.test_seed)

        sampler = DatasetSampler.from_file(setting.problem.name, setting.dataset_path)
        dataset_problem_size = sampler.num_nodes - 1
        if dataset_problem_size != setting.problem.size:
            raise ValueError(
                f"Dataset instances have size {dataset_problem_size} but problem.size is "
                f"{setting.problem.size}."
            )
        if setting.num_test_instances > len(sampler):
            raise ValueError(
                f"num_test_instances={setting.num_test_instances} exceeds the "
                f"{len(sampler)} instances available in {setting.dataset_path}."
            )
        return sampler

    def _evaluate_batch(
        self, method: Method, instance: BaseProblemInstance, batch_idx: int
    ) -> BatchEvaluationResult:
        num_steps = method.search.num_improvement_steps
        step_labels = list(range(num_steps + 1))
        objective_per_step: list[torch.Tensor] = []
        cumulative_time_per_step: list[float] = []

        # Per-step timing forces a sync each step, a small overhead absent in deployment;
        # every method is measured identically, so relative obj(time) comparison stays fair.
        batch_start = time.perf_counter()

        def on_step(step: int, solution: BaseSolution) -> None:
            objective = solution.objective().detach().cpu()  # forces the step's GPU work
            if self.device is not None and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            cumulative_time_per_step.append(time.perf_counter() - batch_start)
            assert objective.ndim == 1, (
                f"Expected 1-D objective tensor, got shape {objective.shape}"
            )
            objective_per_step.append(objective)
            for callback in self.on_step:
                callback(batch_idx, step, solution)

        with torch.no_grad():
            solve(method, instance, on_step=on_step, step_progress=self.step_progress)

        assert len(objective_per_step) == len(step_labels), (
            f"Expected {len(step_labels)} captures, got {len(objective_per_step)}"
        )
        return BatchEvaluationResult(
            step_labels=step_labels,
            objective_per_step=torch.stack(objective_per_step, dim=0),
            cumulative_time_per_step=cumulative_time_per_step,
        )

    def _aggregate(
        self, batches: list[BatchEvaluationResult], *, total_time_s: float
    ) -> EvaluationResult:
        assert batches, "Expected at least one batch"
        step_labels = batches[0].step_labels
        for batch in batches[1:]:
            assert batch.step_labels == step_labels, "Expected all batches to share step_labels"
        per_instance_objective = torch.cat([batch.objective_per_step for batch in batches], dim=1)
        # Total wall-clock to bring every instance to step k, summed over sequential batches.
        time_per_step = [
            sum(batch.cumulative_time_per_step[step] for batch in batches)
            for step in range(len(step_labels))
        ]
        return EvaluationResult(
            step_labels=step_labels,
            objective_per_step=per_instance_objective,
            time_per_step=time_per_step,
            total_time_s=total_time_s,
        )


class PlotSnapshotCallback:
    """Captures per-step solution snapshots from the first batch only."""

    def __init__(self, *, num_plot_instances: int) -> None:
        self.num_plot_instances = num_plot_instances
        self.snapshots: list[list[BaseSolution]] = []

    def on_step(self, batch_idx: int, step: int, solution: BaseSolution) -> None:
        if batch_idx != 0:
            return
        if step == 0:
            batch_size = solution.batch_size[0] if solution.batch_size else 1
            k = min(self.num_plot_instances, batch_size)
            if self.num_plot_instances > batch_size:
                warnings.warn(
                    f"PlotSnapshotCallback: num_plot_instances={self.num_plot_instances} "
                    f"> first batch_size={batch_size}; capturing only {k} snapshots.",
                )
            self.snapshots = [[] for _ in range(k)]
        for local_idx in range(len(self.snapshots)):
            self.snapshots[local_idx].append(solution[local_idx])
