from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from core.eval.config import EvaluationRunConfig
from core.eval.problem_setting import SyntheticProblemSetting
from core.evaluation import BatchCallback, BatchEvaluationResult
from core.methods.nds_trained import NDSTrainedConfig
from core.models.metric import MeanMetric, MetricCollection, SumMetric
from core.results.row import SummaryRowBase
from core.results.synthetic import SyntheticStepRow


@contextmanager
def eval_run_logger(run: EvaluationRunConfig, *, enabled: bool) -> Iterator[EvalRunLogger]:
    """Scope a single setting's wandb run.

    Initialises the wandb run on enter and finalises it on exit, emitting whatever
    ``summary_row``/``curve`` the caller recorded on the logger. When ``enabled`` is
    ``False`` every method is a no-op and no wandb run is created.
    """
    logger = EvalRunLogger(run, enabled=enabled)
    logger._init()
    try:
        yield logger
    finally:
        logger._finalize()


class EvalRunLogger:
    """Logs one evaluation setting to wandb across its batch/instance lifecycle."""

    def __init__(self, run: EvaluationRunConfig, *, enabled: bool) -> None:
        self._run = run
        self._enabled = enabled
        self.summary_row: SummaryRowBase | None = None
        self.curve: list[SyntheticStepRow] | None = None
        self._cvrplib_gap = MeanMetric()

    def batch_callbacks(
        self,
        setting: SyntheticProblemSetting,
        *,
        num_improvement_steps: int,
        checkpoint_steps: list[int],
    ) -> list[BatchCallback] | None:
        """Per-batch callback logging running synthetic metrics, or ``None`` when disabled."""
        if not self._enabled:
            return None
        checkpoints = [step for step in checkpoint_steps if step <= num_improvement_steps]
        batch_logger = _WandbBatchLogger(setting.label, setting.num_test_instances, checkpoints)
        return [batch_logger.on_batch_end]

    def log_cvrplib_instance(self, label: str, *, gap_pct: float, progress: float) -> None:
        """Log a single CVRPLIB instance's gap and the running mean gap."""
        if not self._enabled:
            return
        import wandb

        self._cvrplib_gap.update(gap_pct)
        wandb.log(
            {
                f"eval/{label}/instance/gap_pct": gap_pct,
                f"eval/{label}/running/gap_pct": self._cvrplib_gap.compute(),
                f"eval/{label}/progress": progress,
            }
        )

    def _init(self) -> None:
        if not self._enabled:
            return
        import wandb

        run = self._run
        setting = run.setting
        trained = isinstance(run.method, NDSTrainedConfig)
        source_run_name = Path(run.method.run_dir).name if trained else None
        if isinstance(setting, SyntheticProblemSetting):
            problem = {"name": setting.problem.name, "size": setting.problem.size}
        else:
            problem = {"name": run.problem_name, "set_name": setting.set_name}
        tags = [run.method.type]
        # Construction-only methods run no destroy/repair search, so have no M.
        search = getattr(run.method, "search", None)
        if search is not None:
            tags.append(f"M={search.deconstruction_size}")
        tags += [run.group, setting.label]
        wandb.init(
            project=run.logging.wandb_project,
            name=f"eval-{run.name}-{setting.label}",
            job_type="eval",
            group=source_run_name if trained else run.name,
            tags=tags,
            config={
                "problem": problem,
                "group": run.group,
                "setting": setting.label,
                "eval_run": run.model_dump(mode="json"),
            },
            reinit=True,
        )

    def _finalize(self) -> None:
        if not self._enabled:
            return
        import wandb

        label = self._run.setting.label
        if self.summary_row is not None:
            for key, value in self.summary_row.model_dump().items():
                if isinstance(value, (int, float)) and key not in {"n_instances", "size"}:
                    wandb.summary[f"eval/{label}/{key}"] = value
        if self.curve is not None:
            table = wandb.Table(
                columns=["step", "obj_mean", "obj_std", "time_total_s", "time_per_instance_s"],
                data=[
                    [r.step, r.obj_mean, r.obj_std, r.time_total_s, r.time_per_instance_s]
                    for r in self.curve
                ],
            )
            wandb.log(
                {
                    f"eval/{label}/obj_vs_step": wandb.plot.line(
                        table, "step", "obj_mean", title=f"{label} obj vs step"
                    ),
                    f"eval/{label}/obj_vs_time": wandb.plot.line(
                        table, "time_total_s", "obj_mean", title=f"{label} obj vs time"
                    ),
                }
            )
        wandb.finish()


class _WandbBatchLogger:
    """Accumulates running synthetic metrics and logs them once per batch."""

    def __init__(self, label: str, total_instances: int, checkpoint_steps: list[int]) -> None:
        self._label = label
        self._total = total_instances
        self._checkpoints = checkpoint_steps
        self._seen = 0
        self._running = MetricCollection(
            {
                "obj_initial": MeanMetric(),
                "obj_final": MeanMetric(),
                "improvement": MeanMetric(),
                "relative_improvement": MeanMetric(),
            }
        )
        self._running_time_total = SumMetric()  # whole-search time, summed over batches
        self._running_obj = {step: MeanMetric() for step in checkpoint_steps}
        self._running_time = {step: SumMetric() for step in checkpoint_steps}

    def on_batch_end(self, _batch_idx: int, batch_result: BatchEvaluationResult) -> None:
        import wandb

        objective = batch_result.objective_per_step
        times = batch_result.cumulative_time_per_step
        batch_size = objective.shape[1]
        self._seen += batch_size

        initial = objective[0].mean().item()
        final = objective[-1].mean().item()
        improvement = initial - final
        relative_improvement = improvement / initial if initial else 0.0
        batch_time_s = times[-1]
        instances_per_sec = batch_size / batch_time_s if batch_time_s > 0.0 else 0.0

        self._running.update(
            {
                "obj_initial": initial,
                "obj_final": final,
                "improvement": improvement,
                "relative_improvement": relative_improvement,
            },
            weight=batch_size,
        )
        self._running_time_total.update(batch_time_s)
        running = self._running.compute()
        total_time = self._running_time_total.compute()

        log: dict[str, float] = {
            f"eval/{self._label}/progress": self._seen / max(self._total, 1),
            f"eval/{self._label}/running/instances": self._seen,
            f"eval/{self._label}/batch/obj_initial": initial,
            f"eval/{self._label}/batch/obj_final": final,
            f"eval/{self._label}/batch/improvement": improvement,
            f"eval/{self._label}/batch/relative_improvement": relative_improvement,
            f"eval/{self._label}/batch/batch_time_s": batch_time_s,
            f"eval/{self._label}/batch/instances_per_sec": instances_per_sec,
            f"eval/{self._label}/running/obj_final": running["obj_final"],
            f"eval/{self._label}/running/improvement": running["improvement"],
            f"eval/{self._label}/running/relative_improvement": running["relative_improvement"],
            f"eval/{self._label}/running/instances_per_sec": self._seen / total_time
            if total_time > 0.0
            else 0.0,
        }
        for step in self._checkpoints:
            batch_obj = objective[step].mean().item()
            self._running_obj[step].update(batch_obj, weight=batch_size)
            self._running_time[step].update(times[step])
            log[f"eval/{self._label}/batch/obj@{step}"] = batch_obj
            log[f"eval/{self._label}/running/obj@{step}"] = self._running_obj[step].compute()
            log[f"eval/{self._label}/running/time_total_s@{step}"] = self._running_time[
                step
            ].compute()
        wandb.log(log)
