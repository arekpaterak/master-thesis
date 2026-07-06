from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from core.eval.config import EvaluationConfig
from core.eval.loader import load_evaluation_config
from core.eval.problem_setting import CVRPLibProblemSetting, SyntheticProblemSetting
from core.evaluation import EvaluationResult, Evaluator, PlotSnapshotCallback
from core.methods import build_method
from core.methods.base import Method
from core.methods.nds_trained import NDSTrainedConfig
from core.problems import make_problem_config
from core.problems.cvrp.dataset import load_cvrplib_instance, load_cvrplib_solution_bks
from core.results.plots import plot_objective_curve, plot_search_progress
from core.results.schema import (
    CVRPLIB_BUCKET_SCHEMA,
    CVRPLIB_INSTANCE_SCHEMA,
    SUMMARY_SCHEMA,
    SYNTHETIC_CURVE_SCHEMA,
    SYNTHETIC_INSTANCE_SCHEMA,
)
from core.results.summary import (
    cvrplib_bucket_rows,
    cvrplib_summary_row,
    synthetic_curve_rows,
    synthetic_instance_rows,
    synthetic_summary_row,
)
from core.results.writer import eval_output_dir, write_config, write_csv_rows
from core.solve import solve

EXPERIMENTS_DIR = Path(__file__).parent
EVAL_CONFIGS_DIR = EXPERIMENTS_DIR / "configs" / "eval"
CVRPLIB_DIR = EXPERIMENTS_DIR.parent / "data" / "cvrplib"
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "eval"

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(..., help="Eval config YAML (path, or name under configs/eval)."),
    device: str | None = typer.Option(None, help="Torch device (default: cuda if available)."),
    problem: list[str] = typer.Option(
        None, "--problem", "-p", help="Only run these problem settings, by label. Repeatable."
    ),
    first: bool = typer.Option(False, "--first", help="Run only the first (selected) problem."),
) -> None:
    """Evaluate a single method across the problem settings listed in the config."""
    load_dotenv(".env")
    config_path = config if config.exists() else EVAL_CONFIGS_DIR / config
    eval_config = load_evaluation_config(config_path)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    settings = _select_problems(eval_config, problem, first)
    eval_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_dir = eval_output_dir(RESULTS_DIR, eval_config.name, eval_id)

    method = _build_method(eval_config, torch_device)
    labels = ", ".join(setting.label for setting in settings)
    console.print(
        f"[bold]{eval_config.name}[/bold]  |  method: {eval_config.method.type}  |  "
        f"device: {torch_device}  |  {labels}"
    )

    summary_rows: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, object]]] = {}
    # TODO: Upload after each setting
    for setting in settings:
        if isinstance(setting, CVRPLibProblemSetting):
            summary_rows.append(_run_cvrplib(method, setting, eval_config, torch_device, eval_dir))
        else:
            summary_row, curve = _run_synthetic(
                method, setting, eval_config, torch_device, eval_dir
            )
            summary_rows.append(summary_row)
            curves[setting.label] = curve

    summary_rows = [row for row in summary_rows if row is not None]
    write_csv_rows(summary_rows, schema=SUMMARY_SCHEMA, path=eval_dir / "summary.csv")
    write_config(eval_config.model_dump(mode="json"), path=eval_dir / "config.yml")
    console.print(f"Wrote results to [bold]{eval_dir}[/bold]")

    if eval_config.logging.wandb_project:
        _upload_wandb(eval_config, eval_id, eval_dir, summary_rows, curves)

    _release_gpu(method)


def _select_problems(
    eval_config: EvaluationConfig, labels: list[str] | None, first: bool
) -> list[SyntheticProblemSetting | CVRPLibProblemSetting]:
    settings = list(eval_config.problems)
    if labels:
        available = {setting.label for setting in settings}
        unknown = [label for label in labels if label not in available]
        if unknown:
            raise typer.BadParameter(
                f"Unknown problem label(s): {', '.join(unknown)}. Available: {', '.join(available)}"
            )
        wanted = set(labels)
        settings = [setting for setting in settings if setting.label in wanted]
    if first:
        settings = settings[:1]
    if not settings:
        raise typer.BadParameter("No problem settings selected.")
    return settings


def _build_method(eval_config: EvaluationConfig, device: torch.device) -> Method:
    # NDS generalizes across sizes and the baselines only need problem.name, so the
    # method is built once and reused across every problem setting.
    representative_problem = make_problem_config(name=eval_config.problem_name, size=100)
    return build_method(eval_config.method, problem=representative_problem, device=device)


def _run_synthetic(
    method: Method,
    setting: SyntheticProblemSetting,
    eval_config: EvaluationConfig,
    device: torch.device,
    eval_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    console.rule(f"[bold]{setting.label}[/bold] (synthetic)")
    plot_cb = PlotSnapshotCallback(num_plot_instances=eval_config.num_plot_instances)
    evaluator = Evaluator(
        on_step=[plot_cb.on_step] if eval_config.num_plot_instances else None,
        batch_progress=lambda it: tqdm(it, desc="Batches"),
        step_progress=lambda it: tqdm(it, desc="Improvement steps", leave=False),
        device=device,
    )
    result = evaluator.evaluate(setting, method)
    _report_synthetic(result)

    write_csv_rows(
        synthetic_instance_rows(config=eval_config.name, setting=setting, result=result),
        schema=SYNTHETIC_INSTANCE_SCHEMA,
        path=eval_dir / f"{setting.label}_instances.csv",
    )
    curve = synthetic_curve_rows(result)
    write_csv_rows(
        curve, schema=SYNTHETIC_CURVE_SCHEMA, path=eval_dir / f"{setting.label}_curve.csv"
    )
    _save_synthetic_plots(setting, result, plot_cb, eval_dir)

    summary_row = synthetic_summary_row(
        config=eval_config.name, method_type=eval_config.method.type, setting=setting, result=result
    )
    return summary_row, curve


def _save_synthetic_plots(
    setting: SyntheticProblemSetting,
    result: EvaluationResult,
    plot_cb: PlotSnapshotCallback,
    eval_dir: Path,
) -> None:
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mean = result.objective_per_step.mean(dim=1).tolist()
    std = result.objective_per_step.std(dim=1).tolist()
    fig = plot_objective_curve(result.step_labels, mean, std, title=f"{setting.label}: objective")
    fig.savefig(plots_dir / f"{setting.label}_progress.pdf", bbox_inches="tight")
    plt.close(fig)

    for idx, snapshots in enumerate(plot_cb.snapshots):
        fig = plot_search_progress(snapshots, result.step_labels, epoch=0)
        fig.savefig(plots_dir / f"{setting.label}_routes_{idx:02d}.pdf", bbox_inches="tight")
        plt.close(fig)


def _report_synthetic(result: EvaluationResult) -> None:
    objective = result.objective_per_step
    initial, final = objective[0], objective[-1]
    improvement = initial - final

    table = Table(show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Instances", str(objective.shape[1]))
    table.add_row("Steps", str(objective.shape[0] - 1))
    table.add_row("Initial mean", f"{initial.mean().item():.4f}")
    table.add_row("Final mean", f"{final.mean().item():.4f}")
    table.add_row("Improvement mean", f"{improvement.mean().item():.4f}")
    table.add_row("Improvement std", f"{improvement.std().item():.4f}")
    table.add_row("Total time (s)", f"{result.total_time_s:.2f}")
    console.print(table)


def _run_cvrplib(
    method: Method,
    setting: CVRPLibProblemSetting,
    eval_config: EvaluationConfig,
    device: torch.device,
    eval_dir: Path,
) -> dict[str, object] | None:
    console.rule(f"[bold]{setting.label}[/bold] (CVRPLIB)")
    vrp_paths = _cvrplib_paths(setting)
    console.print(f"{len(vrp_paths)} instances selected from set {setting.set_name}.")
    if not vrp_paths:
        return None

    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for vrp_path in tqdm(vrp_paths, desc="Instances"):
        instance = load_cvrplib_instance(vrp_path).unsqueeze(0).to(device)
        bks = load_cvrplib_solution_bks(vrp_path.with_suffix(".sol"))

        with torch.no_grad():
            final_solution = solve(method, instance)

        our_cost = int(final_solution.total_distance(rounded=True).item())
        rows.append(
            {
                "instance": vrp_path.stem,
                "n_customers": instance.num_customers,
                "bks": bks,
                "our_cost": our_cost,
                "gap_pct": round((our_cost - bks) / bks * 100, 2),
            }
        )
    elapsed = time.perf_counter() - start

    _report_cvrplib(pl.DataFrame(rows))
    write_csv_rows(
        rows, schema=CVRPLIB_INSTANCE_SCHEMA, path=eval_dir / f"{setting.label}_instances.csv"
    )
    write_csv_rows(
        cvrplib_bucket_rows(rows),
        schema=CVRPLIB_BUCKET_SCHEMA,
        path=eval_dir / f"{setting.label}_buckets.csv",
    )
    return cvrplib_summary_row(
        config=eval_config.name,
        method_type=eval_config.method.type,
        setting=setting,
        instance_rows=rows,
        total_time_s=elapsed,
    )


def _report_cvrplib(results: pl.DataFrame) -> None:
    columns = ("instance", "n_customers", "bks", "our_cost", "gap_pct")
    table = Table(show_header=True)
    for column in columns:
        table.add_column(column, justify="left" if column == "instance" else "right")
    for row in results.iter_rows(named=True):
        table.add_row(*(str(row[column]) for column in columns))
    console.print(table)

    summary = results.select(
        pl.col("gap_pct").mean().round(3).alias("mean_gap_pct"),
        pl.col("gap_pct").median().round(3).alias("median_gap_pct"),
        pl.len().alias("n_instances"),
    ).row(0, named=True)
    console.print(
        f"[bold]mean gap[/bold] {summary['mean_gap_pct']}%  |  "
        f"median {summary['median_gap_pct']}%  |  n={summary['n_instances']}"
    )


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


def _upload_wandb(
    eval_config: EvaluationConfig,
    eval_id: str,
    eval_dir: Path,
    summary_rows: list[dict[str, object]],
    curves: dict[str, list[dict[str, object]]],
) -> None:
    import wandb
    from core.results.wandb_artifacts import upload_eval_dir

    trained = isinstance(eval_config.method, NDSTrainedConfig)
    source_run_name = Path(eval_config.method.run_dir).name if trained else None
    problem_labels = [str(row["problem_label"]) for row in summary_rows]

    wandb.init(
        project=eval_config.logging.wandb_project,
        name=f"eval-{eval_config.name}-{eval_id}",
        job_type="eval",
        group=source_run_name if trained else eval_config.name,
        tags=[eval_config.method.type, *problem_labels],
        config={"eval_config": eval_config.model_dump(mode="json"), "eval_id": eval_id},
    )
    for row in summary_rows:
        label = row["problem_label"]
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"n_instances", "size"}:
                wandb.summary[f"{label}/{key}"] = value
    for label, curve in curves.items():
        table = wandb.Table(
            columns=["step", "obj_mean", "obj_std"],
            data=[[r["lns_step"], r["obj_mean"], r["obj_std"]] for r in curve],
        )
        wandb.log({f"{label}/progress": wandb.plot.line(table, "step", "obj_mean", title=label)})

    upload_eval_dir(
        run_name=source_run_name or eval_config.name,
        eval_dir=eval_dir,
        eval_id=eval_id,
        metadata={
            "eval_id": eval_id,
            "config": eval_config.name,
            "method_type": eval_config.method.type,
        },
        link_checkpoint_alias="final" if trained else None,
    )
    wandb.finish()


def _release_gpu(method: Method) -> None:
    model = getattr(method.improvement_step, "_model", None)
    if model is not None:
        model.to("cpu")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
