from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from core.eval.config import EvaluationConfig, EvaluationRunConfig
from core.eval.problem_setting import CVRPLibProblemSetting, SyntheticProblemSetting
from core.evaluation import EvaluationResult, Evaluator, PlotSnapshotCallback
from core.methods import build_method
from core.methods.base import Method
from core.methods.nds_trained import NDSTrainedConfig
from core.problems import make_problem_config
from core.problems.cvrp.dataset import (
    cvrplib_instance_paths,
    load_cvrplib_instance,
    load_cvrplib_solution_bks,
)
from core.results.cvrplib import CVRPLibInstanceRow, CVRPLibSummaryRow
from core.results.plots import plot_objective_curve, plot_search_progress, save_figure
from core.results.synthetic import (
    SyntheticInstanceRow,
    SyntheticStepRow,
    SyntheticSummaryRow,
)
from core.results.wandb_eval import EvalRunLogger, eval_run_logger
from core.solve import solve

EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
EVAL_CONFIGS_DIR = EXPERIMENTS_DIR / "configs" / "eval"
CVRPLIB_DIR = EXPERIMENTS_DIR.parent / "data" / "cvrplib"
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "eval"

console = Console()
app = typer.Typer(add_completion=False)

DEFAULT_CHECKPOINT_STEPS = [10, 20, 50, 100]


@app.command()
def main(
    config_arg: Path = typer.Argument(
        None, help="Eval config file (path, or name under configs/eval)."
    ),
    config_option: Path = typer.Option(
        None, "--config", help="Eval config file (path, or name under configs/eval)."
    ),
    device: str | None = typer.Option(None, help="Torch device (default: cuda if available)."),
    setting: list[str] = typer.Option(
        None, "--setting", "-s", help="Only run these settings, by label."
    ),
    group: list[str] = typer.Option(
        None, "--group", "-g", help="Only run settings in these groups (e.g. main)."
    ),
    checkpoint_steps: list[int] = typer.Option(
        DEFAULT_CHECKPOINT_STEPS,
        "--checkpoint-steps",
        help="Improvement steps to report measurements at.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Reevaluate on config settings even if already done for this config."
    ),
    warmup: int = typer.Option(
        3,
        "--cvrplib-warmup",
        help="Untimed warm-up batches before evaluating CVRPLib instances (0 to disable).",
    ),
) -> None:
    """Evaluate a single method across the problem settings listed in the config."""
    if config_arg and config_option:
        raise ValueError("Expected config passed as either argument or option.")
    config = config_arg or config_option
    if config is None:
        raise ValueError("Expected config")

    config_path = (config if config.exists() else EVAL_CONFIGS_DIR / config).resolve()
    os.chdir(PROJECT_ROOT)
    load_dotenv(".env")
    eval_config = EvaluationConfig.from_file(config_path)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    runs = _select_runs(eval_config.runs(), setting, group)
    method_dir = RESULTS_DIR / eval_config.problem_name / eval_config.name
    if not force:
        runs = _drop_evaluated(method_dir, runs)
    if not runs:
        console.print(
            "All selected problem settings already evaluated; nothing to do (use --force to reevaluate)."
        )
        return

    method = _build_method(eval_config, torch_device)
    checkpoint_epoch = (
        eval_config.method.resolved_checkpoint_epoch()
        if isinstance(eval_config.method, NDSTrainedConfig)
        else None
    )
    labels = ", ".join(run.setting.label for run in runs)
    console.print(
        f"[bold]{eval_config.name}[/bold]  |  method: {eval_config.method.type}  |  "
        f"device: {torch_device}  |  {labels}"
    )

    log_wandb = bool(eval_config.logging.wandb_project)
    for run in runs:
        with eval_run_logger(run, enabled=log_wandb) as logger:
            if isinstance(run.setting, CVRPLibProblemSetting):
                logger.summary_row = _run_cvrplib(
                    method, run, torch_device, method_dir, checkpoint_epoch, logger, warmup
                )
            else:
                logger.summary_row, logger.curve = _run_synthetic(
                    method,
                    run,
                    torch_device,
                    method_dir,
                    checkpoint_epoch,
                    checkpoint_steps,
                    logger,
                )

    eval_config.write_yaml(path=method_dir / "config.yml")
    console.print(f"Wrote results to [bold]{method_dir}[/bold]")


def _select_runs(
    runs: list[EvaluationRunConfig],
    labels: list[str] | None,
    groups: list[str] | None,
) -> list[EvaluationRunConfig]:
    if groups:
        available = {run.group for run in runs}
        unknown = [group for group in groups if group not in available]
        if unknown:
            raise typer.BadParameter(
                f"Unknown group(s): {', '.join(unknown)}. Available: {', '.join(available)}"
            )
        wanted = set(groups)
        runs = [run for run in runs if run.group in wanted]
    if labels:
        # Match setting labels case-insensitively (e.g. cvrplib-X == cvrplib-x, TSP100 == tsp100).
        wanted = {label.lower() for label in labels}
        available = {run.setting.label.lower() for run in runs}
        unknown = [label for label in wanted if label not in available]
        if unknown:
            raise typer.BadParameter(
                f"Unknown setting label(s): {', '.join(unknown)}. Available: {', '.join(available)}"
            )
        runs = [run for run in runs if run.setting.label.lower() in wanted]
    if not runs:
        raise typer.BadParameter("No settings selected.")
    return runs


def _drop_evaluated(method_dir: Path, runs: list[EvaluationRunConfig]) -> list[EvaluationRunConfig]:
    """Drop runs whose setting dir (``<method_dir>/<label>/summary.csv``) already exists."""
    remaining: list[EvaluationRunConfig] = []
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
    run: EvaluationRunConfig,
    device: torch.device,
    method_dir: Path,
    checkpoint_epoch: int | None,
    checkpoint_steps: list[int],
    logger: EvalRunLogger,
) -> tuple[SyntheticSummaryRow, list[SyntheticStepRow]]:
    setting = run.setting
    console.rule(f"[bold]{setting.label}[/bold] ({run.group}, synthetic)")
    plot_cb = PlotSnapshotCallback(num_plot_instances=run.num_plot_instances)

    evaluator = Evaluator(
        on_step=[plot_cb.on_step] if run.num_plot_instances else None,
        on_batch_end=logger.batch_callbacks(
            setting,
            num_improvement_steps=method.search.num_improvement_steps,
            checkpoint_steps=checkpoint_steps,
        ),
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

    mean = result.objective_per_step.mean(dim=1).tolist()
    std = result.objective_per_step.std(dim=1).tolist()
    fig = plot_objective_curve(result.step_labels, mean, std, title=f"{setting.label}: objective")
    save_figure(fig, plots_dir / f"{setting.label}_progress.pdf")
    plt.close(fig)

    for idx, snapshots in enumerate(plot_cb.snapshots):
        fig = plot_search_progress(snapshots, result.step_labels, epoch=0)
        save_figure(fig, plots_dir / f"{setting.label}_routes_{idx:02d}.pdf")
        plt.close(fig)


def _run_cvrplib(
    method: Method,
    run: EvaluationRunConfig,
    device: torch.device,
    method_dir: Path,
    checkpoint_epoch: int | None,
    logger: EvalRunLogger,
    warmup: int,
) -> CVRPLibSummaryRow | None:
    setting = run.setting
    console.rule(f"[bold]{setting.label}[/bold] ({run.group}, CVRPLIB)")
    try:
        vrp_paths = cvrplib_instance_paths(
            CVRPLIB_DIR / setting.set_name,
            min_customers=setting.min_customers,
            max_customers=setting.max_customers,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"{len(vrp_paths)} instances selected from set {setting.set_name}.")
    if not vrp_paths:
        return None

    # Untimed warm-up so first-call overhead (lazy init, GPU kernel/JIT caches) does not
    # inflate the per-instance timings of the measured run.
    warmup = min(warmup, len(vrp_paths))
    if warmup:
        console.print(f"Warming up on {warmup} instance(s) before timing.")
        for vrp_path in tqdm(vrp_paths[:warmup], desc="Warm-up", leave=False):
            instance = load_cvrplib_instance(vrp_path).unsqueeze(0).to(device)
            with torch.no_grad():
                solve(method, instance)

    rows: list[CVRPLibInstanceRow] = []
    for idx, vrp_path in enumerate(tqdm(vrp_paths, desc="Instances")):
        instance = load_cvrplib_instance(vrp_path).unsqueeze(0).to(device)
        bks = load_cvrplib_solution_bks(vrp_path.with_suffix(".sol"))

        start = time.perf_counter()
        with torch.no_grad():
            final_solution = solve(method, instance)
        instance_time_s = time.perf_counter() - start

        obj = int(final_solution.total_distance(rounded=True).item())
        gap_pct = round((obj - bks) / bks * 100, 2)
        rows.append(
            CVRPLibInstanceRow(
                instance=vrp_path.stem,
                n_customers=int(instance.num_customers),
                bks=bks,
                obj=obj,
                gap_pct=gap_pct,
                time_s=round(instance_time_s, 4),
            )
        )
        logger.log_cvrplib_instance(
            setting.label, gap_pct=gap_pct, progress=(idx + 1) / len(vrp_paths)
        )

    setting_dir = method_dir / setting.label
    CVRPLibInstanceRow.write(rows, path=setting_dir / "instances.csv")
    summary_rows = CVRPLibSummaryRow.build(
        config=run.name,
        method_type=run.method.type,
        group=run.group,
        checkpoint_epoch=checkpoint_epoch,
        setting=setting,
        instance_rows=rows,
        destruction_size=method.search.deconstruction_size,
    )
    CVRPLibSummaryRow.write(summary_rows, path=setting_dir / "summary.csv")
    return next((row for row in summary_rows if row.customer_range == "all"), None)


if __name__ == "__main__":
    app()
