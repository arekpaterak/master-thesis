from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from core.eval.config import EvalRunConfig, EvaluationConfig
from core.eval.problem_setting import CVRPLibProblemSetting, SyntheticProblemSetting
from core.evaluation import (
    BatchEvaluationResult,
    EvaluationResult,
    Evaluator,
    PlotSnapshotCallback,
)
from core.methods import build_method
from core.methods.base import Method
from core.methods.nds_trained import NDSTrainedConfig
from core.models.metric import MeanMetric, MetricCollection, SumMetric
from core.problems import make_problem_config
from core.problems.cvrp.dataset import load_cvrplib_instance, load_cvrplib_solution_bks
from core.results.cvrplib import CvrplibInstanceRow, CvrplibSummaryRow
from core.results.plots import plot_objective_curve, plot_search_progress
from core.results.row import SummaryRowBase
from core.results.synthetic import (
    SyntheticInstanceRow,
    SyntheticStepRow,
    SyntheticSummaryRow,
)
from core.solve import solve

EXPERIMENTS_DIR = Path(__file__).parent
EVAL_CONFIGS_DIR = EXPERIMENTS_DIR / "configs" / "eval"
CVRPLIB_DIR = EXPERIMENTS_DIR.parent / "data" / "cvrplib"
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "eval"

console = Console()
app = typer.Typer(add_completion=False)

DEFAULT_CHECKPOINT_STEPS = [10, 20, 50, 100]


@app.command()
def main(
    config: Path = typer.Option(..., help="Eval config file (path, or name under configs/eval)."),
    device: str | None = typer.Option(None, help="Torch device (default: cuda if available)."),
    setting: list[str] = typer.Option(
        None, "--setting", "-s", help="Only run these settings, by label."
    ),
    first: bool = typer.Option(False, "--first", help="Run only the first (selected) setting."),
    checkpoint_steps: list[int] = typer.Option(
        DEFAULT_CHECKPOINT_STEPS,
        "--checkpoint-steps",
        help="Improvement steps to report obj, time at.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Recompute settings even if already evaluated for this config."
    ),
) -> None:
    """Evaluate a single method across the settings listed in the config."""
    load_dotenv(".env")
    config_path = config if config.exists() else EVAL_CONFIGS_DIR / config
    eval_config = EvaluationConfig.from_file(config_path)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    runs = _select_runs(eval_config.runs(), setting, first)
    method_dir = RESULTS_DIR / eval_config.problem_name / eval_config.name
    if not force:
        runs = _drop_evaluated(method_dir, runs)
    if not runs:
        console.print("All selected settings already evaluated; nothing to do (use --force).")
        return

    method = _build_method(eval_config, torch_device)
    checkpoint_epoch = _checkpoint_epoch(eval_config.method)
    labels = ", ".join(run.setting.label for run in runs)
    console.print(
        f"[bold]{eval_config.name}[/bold]  |  method: {eval_config.method.type}  |  "
        f"device: {torch_device}  |  {labels}"
    )

    log_wandb = bool(eval_config.logging.wandb_project)

    synthetic_summaries: list[SyntheticSummaryRow] = []
    cvrplib_summaries: list[CvrplibSummaryRow] = []
    for run in runs:
        # One wandb run per setting, so each setting's metrics and artifact live under
        # their own run id.
        if log_wandb:
            _init_run_wandb(run)

        curve: list[SyntheticStepRow] | None = None
        if isinstance(run.setting, CVRPLibProblemSetting):
            summary_row = _run_cvrplib(
                method, run, torch_device, method_dir, checkpoint_epoch, log_wandb
            )
            if summary_row is not None:
                cvrplib_summaries.append(summary_row)
        else:
            summary_row, curve = _run_synthetic(
                method, run, torch_device, method_dir, checkpoint_epoch, checkpoint_steps, log_wandb
            )
            synthetic_summaries.append(summary_row)

        if log_wandb:
            _finalize_run_wandb(run, method_dir, summary_row, curve)

    eval_config.write_yaml(path=method_dir / "config.yml")
    console.print(f"Wrote results to [bold]{method_dir}[/bold]")


class WandbBatchLogger:
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


def _checkpoint_epoch(method_config: object) -> int | None:
    """Epoch of the checkpoint a trained method loads (None for untrained methods)."""
    if not isinstance(method_config, NDSTrainedConfig):
        return None
    checkpoints = sorted((method_config.run_dir / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints:
        return None
    return int(checkpoints[-1].stem.split("_")[1])


def _select_runs(
    runs: list[EvalRunConfig], labels: list[str] | None, first: bool
) -> list[EvalRunConfig]:
    if labels:
        available = {run.setting.label for run in runs}
        unknown = [label for label in labels if label not in available]
        if unknown:
            raise typer.BadParameter(
                f"Unknown setting label(s): {', '.join(unknown)}. Available: {', '.join(available)}"
            )
        wanted = set(labels)
        runs = [run for run in runs if run.setting.label in wanted]
    if first:
        runs = runs[:1]
    if not runs:
        raise typer.BadParameter("No settings selected.")
    return runs


def _drop_evaluated(method_dir: Path, runs: list[EvalRunConfig]) -> list[EvalRunConfig]:
    """Drop runs whose setting dir (``<method_dir>/<label>/summary.csv``) already exists."""
    remaining: list[EvalRunConfig] = []
    for run in runs:
        if (method_dir / run.setting.label / "summary.csv").exists():
            console.print(
                f"[dim]{run.setting.label}: already evaluated; skipping (use --force to "
                "recompute).[/dim]"
            )
        else:
            remaining.append(run)
    return remaining


def _build_method(eval_config: EvaluationConfig, device: torch.device) -> Method:
    # NDS generalizes across sizes and the baselines only need problem.name, so the
    # method is built once and reused across every setting.
    representative_problem = make_problem_config(name=eval_config.problem_name, size=100)
    return build_method(eval_config.method, problem=representative_problem, device=device)


def _run_synthetic(
    method: Method,
    run: EvalRunConfig,
    device: torch.device,
    method_dir: Path,
    checkpoint_epoch: int | None,
    checkpoint_steps: list[int],
    log_wandb: bool,
) -> tuple[SyntheticSummaryRow, list[SyntheticStepRow]]:
    setting = run.setting
    console.rule(f"[bold]{setting.label}[/bold] ({run.group}, synthetic)")
    plot_cb = PlotSnapshotCallback(num_plot_instances=run.num_plot_instances)

    on_batch_end = None
    if log_wandb:
        num_steps = method.search.num_improvement_steps
        checkpoints = [step for step in checkpoint_steps if step <= num_steps]
        batch_logger = WandbBatchLogger(setting.label, setting.num_test_instances, checkpoints)
        on_batch_end = [batch_logger.on_batch_end]

    evaluator = Evaluator(
        on_step=[plot_cb.on_step] if run.num_plot_instances else None,
        on_batch_end=on_batch_end,
        batch_progress=lambda it: tqdm(it, desc="Batches"),
        step_progress=lambda it: tqdm(it, desc="Improvement steps", leave=False),
        device=device,
    )
    result = evaluator.evaluate(setting, method)

    setting_dir = method_dir / setting.label
    SyntheticInstanceRow.write(
        SyntheticInstanceRow.build(
            config=run.name, group=run.group, setting=setting, result=result
        ),
        path=setting_dir / "instances.csv",
    )
    curve = SyntheticStepRow.build(result)
    SyntheticStepRow.write(curve, path=setting_dir / "steps.csv")
    _save_synthetic_plots(setting, result, plot_cb, setting_dir)

    summary_row = SyntheticSummaryRow.build(
        config=run.name,
        method_type=run.method.type,
        group=run.group,
        checkpoint_epoch=checkpoint_epoch,
        setting=setting,
        result=result,
        destruction_size=method.search.deconstruction_size,
    )
    SyntheticSummaryRow.write([summary_row], path=setting_dir / "summary.csv")
    return summary_row, curve


def _save_synthetic_plots(
    setting: SyntheticProblemSetting,
    result: EvaluationResult,
    plot_cb: PlotSnapshotCallback,
    setting_dir: Path,
) -> None:
    plots_dir = setting_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mean = result.objective_per_step.mean(dim=1).tolist()
    std = result.objective_per_step.std(dim=1).tolist()
    fig = plot_objective_curve(result.step_labels, mean, std, title=f"{setting.label}: objective")
    fig.savefig(
        plots_dir / f"{setting.label}_progress.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)

    for idx, snapshots in enumerate(plot_cb.snapshots):
        fig = plot_search_progress(snapshots, result.step_labels, epoch=0)
        fig.savefig(
            plots_dir / f"{setting.label}_routes_{idx:02d}.pdf",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
        )
        plt.close(fig)


def _run_cvrplib(
    method: Method,
    run: EvalRunConfig,
    device: torch.device,
    method_dir: Path,
    checkpoint_epoch: int | None,
    log_wandb: bool,
) -> CvrplibSummaryRow | None:
    setting = run.setting
    console.rule(f"[bold]{setting.label}[/bold] ({run.group}, CVRPLIB)")
    vrp_paths = _cvrplib_paths(setting)
    console.print(f"{len(vrp_paths)} instances selected from set {setting.set_name}.")
    if not vrp_paths:
        return None

    rows: list[CvrplibInstanceRow] = []
    running_gap = MeanMetric()
    start = time.perf_counter()
    for idx, vrp_path in enumerate(tqdm(vrp_paths, desc="Instances")):
        instance = load_cvrplib_instance(vrp_path).unsqueeze(0).to(device)
        bks = load_cvrplib_solution_bks(vrp_path.with_suffix(".sol"))

        with torch.no_grad():
            final_solution = solve(method, instance)

        our_cost = int(final_solution.total_distance(rounded=True).item())
        gap_pct = round((our_cost - bks) / bks * 100, 2)
        rows.append(
            CvrplibInstanceRow(
                instance=vrp_path.stem,
                n_customers=int(instance.num_customers),
                bks=bks,
                our_cost=our_cost,
                gap_pct=gap_pct,
            )
        )
        if log_wandb:
            import wandb

            running_gap.update(gap_pct)
            wandb.log(
                {
                    f"eval/{setting.label}/instance/gap_pct": gap_pct,
                    f"eval/{setting.label}/running/gap_pct": running_gap.compute(),
                    f"eval/{setting.label}/progress": (idx + 1) / len(vrp_paths),
                }
            )
    elapsed = time.perf_counter() - start

    setting_dir = method_dir / setting.label
    CvrplibInstanceRow.write(rows, path=setting_dir / "instances.csv")
    summary_row = CvrplibSummaryRow.build(
        config=run.name,
        method_type=run.method.type,
        group=run.group,
        checkpoint_epoch=checkpoint_epoch,
        setting=setting,
        instance_rows=rows,
        total_time_s=elapsed,
        destruction_size=method.search.deconstruction_size,
    )
    CvrplibSummaryRow.write([summary_row], path=setting_dir / "summary.csv")
    return summary_row


def _cvrplib_paths(setting: CVRPLibProblemSetting) -> list[Path]:
    set_dir = CVRPLIB_DIR / setting.set_name
    if not set_dir.exists():
        raise typer.BadParameter(f"CVRPLIB set dir not found: {set_dir}")
    return [
        path
        for path in sorted(set_dir.glob("*.vrp"))
        if setting.min_customers <= _n_customers_from_stem(path.stem) <= setting.max_customers
    ]


def _n_customers_from_stem(stem: str) -> int:
    # e.g. "X-n101-k25" -> 100
    try:
        return int(stem.split("-n")[1].split("-")[0]) - 1
    except (IndexError, ValueError):
        return -1


def _init_run_wandb(run: EvalRunConfig) -> None:
    """Start a per-run wandb run before evaluating ``run.setting`` so metrics stream live.

    Every run of an eval shares a wandb ``group`` (the source run or config name) so they
    stay together in the dashboard while keeping individual run ids. A top-level
    ``problem`` block and ``group`` are logged so ``problem.name`` / ``problem.size`` /
    ``group`` are filterable, like train.
    """
    import wandb

    setting = run.setting
    trained = isinstance(run.method, NDSTrainedConfig)
    source_run_name = Path(run.method.run_dir).name if trained else None
    if isinstance(setting, SyntheticProblemSetting):
        problem = {"name": setting.problem.name, "size": setting.problem.size}
    else:
        problem = {"name": run.problem_name, "set_name": setting.set_name}
    wandb.init(
        project=run.logging.wandb_project,
        name=f"eval-{run.name}-{setting.label}",
        job_type="eval",
        group=source_run_name if trained else run.name,
        tags=[
            run.method.type,
            f"M={run.method.search.deconstruction_size}",
            run.group,
            setting.label,
        ],
        config={
            "problem": problem,
            "group": run.group,
            "setting": setting.label,
            "eval_run": run.model_dump(mode="json"),
        },
        reinit=True,
    )


def _finalize_run_wandb(
    run: EvalRunConfig,
    method_dir: Path,
    summary_row: SummaryRowBase | None,
    curve: list[SyntheticStepRow] | None,
) -> None:
    """Log this run's summary scalars and obj curves, upload its setting dir, then finish.

    The artifact is logged to the active per-run run, so it lives under that run's id.
    """
    import wandb
    from core.results.wandb_artifacts import upload_eval_dir

    label = run.setting.label
    if summary_row is not None:
        for key, value in summary_row.model_dump().items():
            if isinstance(value, (int, float)) and key not in {"n_instances", "size"}:
                wandb.summary[f"eval/{label}/{key}"] = value
    if curve is not None:
        table = wandb.Table(
            columns=["step", "obj_mean", "obj_std", "time_total_s", "time_per_instance_s"],
            data=[
                [r.step, r.obj_mean, r.obj_std, r.time_total_s, r.time_per_instance_s]
                for r in curve
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

    trained = isinstance(run.method, NDSTrainedConfig)
    source_run_name = Path(run.method.run_dir).name if trained else None
    upload_eval_dir(
        run_name=f"{source_run_name or run.name}-{label}",
        eval_dir=method_dir / label,
        metadata={
            "config": run.name,
            "group": run.group,
            "method_type": run.method.type,
            "setting": label,
        },
    )
    wandb.finish()


if __name__ == "__main__":
    app()
