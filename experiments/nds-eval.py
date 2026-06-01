import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")

with app.setup(hide_code=True):
    import sys
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import yaml
    from dotenv import load_dotenv

    import wandb
    from core.models.nds.config import TrainConfig
    from core.models.nds.eval_config import EvaluationConfig

    load_dotenv(".env")

    RUNS_DIR = Path(__file__).parent / "runs"
    EVAL_CONFIGS_DIR = Path(__file__).parent / "configs" / "eval"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NDS Evaluation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return (device,)


@app.cell(hide_code=True)
def _():
    from wigglystuff import EnvConfig

    is_script_mode = mo.app_meta().mode == "script"

    env_config_widget = mo.ui.anywidget(
        EnvConfig(
            {
                "WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True),
            }
        )
    )

    cli_run_name: str | None = None
    cli_eval_yaml: dict = {}
    if is_script_mode:
        _cli_args = mo.cli_args()

        if not _cli_args or _cli_args.get("help"):
            from rich.console import Console
            from rich.table import Table

            _usage = Table(title="NDS Evaluation -- CLI Usage")
            _usage.add_column("Flag", style="cyan")
            _usage.add_column("Description")
            _usage.add_row(
                "--run <run_name>", "Training run to evaluate (dir under experiments/runs/)."
            )
            _usage.add_row("--eval-config <path>", "YAML preset for EvaluationConfig.")
            _usage.add_row("--<field> <value>", "Override a single EvaluationConfig field.")
            Console().print(_usage)

            _defaults = EvaluationConfig()
            _fields = Table(title="EvaluationConfig fields and defaults")
            _fields.add_column("Key", style="cyan")
            _fields.add_column("Default", style="yellow")
            for _field, _value in _defaults.model_dump().items():
                _fields.add_row(_field, str(_value))
            Console().print(_fields)

            sys.exit(0)

        cli_run_name = str(_cli_args.get("run")) if _cli_args.get("run") else None
        _eval_config_path = _cli_args.get("eval-config") or _cli_args.get("eval_config")
        if _eval_config_path:
            with open(str(_eval_config_path)) as _f:
                cli_eval_yaml = yaml.safe_load(_f) or {}

        for _key, _value in _cli_args.items():
            if _key in ("run", "eval-config", "eval_config", "help"):
                continue
            cli_eval_yaml[_key] = _value
    return cli_eval_yaml, cli_run_name, env_config_widget, is_script_mode


@app.cell(hide_code=True)
def _(env_config_widget, is_script_mode):
    env_config_widget if not is_script_mode and not wandb.login() else None
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Run selection
    """)
    return


@app.cell(hide_code=True)
def _():
    def list_completed_runs() -> list[str]:
        if not RUNS_DIR.exists():
            return []
        runs = []
        for run_dir in RUNS_DIR.glob("nds-*"):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "config.yml").exists():
                continue
            checkpoints = list((run_dir / "checkpoints").glob("epoch_*.pt"))
            if checkpoints:
                runs.append(run_dir.name)
        return sorted(runs)

    completed_runs = list_completed_runs()
    return (completed_runs,)


@app.cell(hide_code=True)
def _(cli_run_name, completed_runs, is_script_mode):
    if is_script_mode:
        if cli_run_name is None:
            raise ValueError("Pass --run <run_name> in script mode.")
        chosen_run_name = cli_run_name
    else:
        if not completed_runs:
            mo.stop(True, mo.md("**No completed runs found under `experiments/runs/`.**"))
        run_dropdown = mo.ui.dropdown(
            completed_runs, value=completed_runs[-1], label="Training run"
        )
        run_dropdown
        chosen_run_name = run_dropdown.value
    return (chosen_run_name,)


@app.cell(hide_code=True)
def _(chosen_run_name):
    run_dir = RUNS_DIR / chosen_run_name
    if not run_dir.exists():
        raise ValueError(f"Run dir not found: {run_dir}")
    config_yml_text = (run_dir / "config.yml").read_text()
    train_config = TrainConfig.model_validate(yaml.safe_load(config_yml_text))
    return config_yml_text, run_dir, train_config


@app.cell(hide_code=True)
def _(config_yml_text):
    mo.md(f"""
### Training config

```yaml
{config_yml_text}
```
""")
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation config
    """)
    return


@app.cell(hide_code=True)
def _():
    presets = (
        ["(defaults)"] + sorted(p.name for p in EVAL_CONFIGS_DIR.glob("*.yaml"))
        if EVAL_CONFIGS_DIR.exists()
        else ["(defaults)"]
    )
    return (presets,)


@app.cell(hide_code=True)
def _(is_script_mode, presets):
    if is_script_mode:
        preset_dropdown_value = "(defaults)"
    else:
        preset_dropdown = mo.ui.dropdown(presets, value=presets[0], label="Eval preset")
        preset_dropdown
        preset_dropdown_value = preset_dropdown.value
    return (preset_dropdown_value,)


@app.cell(hide_code=True)
def _(cli_eval_yaml, is_script_mode, preset_dropdown_value):
    if is_script_mode:
        preset_yaml = dict(cli_eval_yaml)
    elif preset_dropdown_value == "(defaults)":
        preset_yaml = {}
    else:
        with open(EVAL_CONFIGS_DIR / preset_dropdown_value) as _f:
            preset_yaml = yaml.safe_load(_f) or {}
    preset_defaults = EvaluationConfig.model_validate(preset_yaml)
    return (preset_defaults,)


@app.cell(hide_code=True)
def _(preset_defaults):
    num_test_instances_w = mo.ui.number(
        1, 10000, value=preset_defaults.num_test_instances, step=1, label="Num test instances"
    )
    test_seed_w = mo.ui.number(
        0, 2**31 - 1, value=preset_defaults.test_seed, step=1, label="Test seed"
    )
    num_improvement_steps_w = mo.ui.number(
        1, 5000, value=preset_defaults.num_improvement_steps, step=10, label="LNS steps"
    )
    rollouts_per_step_w = mo.ui.number(
        1, 1000, value=preset_defaults.rollouts_per_step, step=10, label="Rollouts/step"
    )
    n_repairs_w = mo.ui.number(
        1, 100, value=preset_defaults.n_repairs, step=1, label="Repairs/rollout"
    )
    temperature_start_w = mo.ui.number(
        0.0, 10.0, value=preset_defaults.temperature_start, step=0.01, label="SA start temp"
    )
    temperature_min_w = mo.ui.number(
        0.0, 10.0, value=preset_defaults.temperature_min, step=0.001, label="SA min temp"
    )
    num_plot_instances_w = mo.ui.number(
        0, 100, value=preset_defaults.num_plot_instances, step=1, label="Plot first K instances"
    )
    method_suffix_w = mo.ui.text(
        value=preset_defaults.method_suffix or "", label="Method suffix (optional)"
    )
    mo.vstack(
        [
            num_test_instances_w,
            test_seed_w,
            num_improvement_steps_w,
            rollouts_per_step_w,
            n_repairs_w,
            temperature_start_w,
            temperature_min_w,
            num_plot_instances_w,
            method_suffix_w,
        ]
    )
    return (
        method_suffix_w,
        n_repairs_w,
        num_improvement_steps_w,
        num_plot_instances_w,
        num_test_instances_w,
        rollouts_per_step_w,
        temperature_min_w,
        temperature_start_w,
        test_seed_w,
    )


@app.cell(hide_code=True)
def _(
    is_script_mode,
    method_suffix_w,
    n_repairs_w,
    num_improvement_steps_w,
    num_plot_instances_w,
    num_test_instances_w,
    preset_defaults,
    rollouts_per_step_w,
    temperature_min_w,
    temperature_start_w,
    test_seed_w,
):
    if is_script_mode:
        eval_config = preset_defaults
    else:
        eval_config = EvaluationConfig(
            num_test_instances=num_test_instances_w.value,
            test_seed=test_seed_w.value,
            num_improvement_steps=num_improvement_steps_w.value,
            rollouts_per_step=rollouts_per_step_w.value,
            n_repairs=n_repairs_w.value,
            temperature_start=temperature_start_w.value,
            temperature_min=temperature_min_w.value,
            num_plot_instances=num_plot_instances_w.value,
            method_suffix=(method_suffix_w.value.strip() or None),
        )
    return (eval_config,)


@app.cell(hide_code=True)
def _(is_script_mode):
    run_eval_button = mo.ui.run_button(label="Run evaluation")
    run_eval_button if not is_script_mode else None
    return (run_eval_button,)


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    ## Execute
    """)
    return


@app.cell
def _(
    chosen_run_name,
    device,
    eval_config,
    is_script_mode,
    run_dir,
    run_eval_button,
    train_config,
):
    if not is_script_mode:
        mo.stop(not run_eval_button.value, mo.md("Click 'Run evaluation' to start."))

    from core.models.nds.evaluation import configure_evaluation_step, plot_lns_progress
    from core.models.nds.factory import build_model
    from core.problems.setup import get_start_solution_and_repair, make_sampler
    from core.results.wandb_artifacts import upload_eval_plots, upload_eval_results
    from core.results.writer import write_eval_results

    def find_latest_checkpoint(dir_: Path) -> tuple[Path | None, int]:
        ckpts = sorted((dir_ / "checkpoints").glob("epoch_*.pt"))
        if not ckpts:
            return None, 0
        return ckpts[-1], int(ckpts[-1].stem.split("_")[1])

    model = build_model(train_config).to(device)
    final_ckpt, final_epoch = find_latest_checkpoint(run_dir)
    if final_ckpt is None:
        raise ValueError(f"No checkpoint found under {run_dir / 'checkpoints'}.")
    _ckpt = torch.load(final_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(_ckpt["model"])
    model.eval()

    test_sampler = make_sampler(
        train_config.problem.name, train_config.problem.size, seed=eval_config.test_seed
    )
    generate_start, repair_fn, apply_rollout_fn = get_start_solution_and_repair(
        train_config.problem.name, noop=False
    )

    eval_step = configure_evaluation_step(
        generate_start_solution=generate_start,
        apply_rollout_fn=apply_rollout_fn,
        repair=repair_fn,
        deconstruction_size=train_config.search.deconstruction_size,
        eval_config=eval_config,
    )

    instance = test_sampler(eval_config.num_test_instances)
    result = eval_step(model, instance)

    method = f"nds-{train_config.model.decoder.type}"
    if eval_config.method_suffix:
        method = f"{method}-{eval_config.method_suffix}"

    eval_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    problem_tag = f"{train_config.problem.name}{train_config.problem.size}"

    rows = [
        {
            "method": method,
            "instance_id": f"{problem_tag}-seed{eval_config.test_seed}-{i:04d}",
            "seed": eval_config.test_seed,
            "lns_step": step,
            "objective": result.per_instance_objective[step_idx][i],
            "time_s": result.cumulative_time_s[step_idx],
            "initial_objective": result.per_instance_objective[0][i],
        }
        for step_idx, step in enumerate(result.step_labels)
        for i in range(eval_config.num_test_instances)
    ]

    parquet_path = write_eval_results(run_dir, eval_run_id=eval_id, rows=rows)

    plots_dir = run_dir / "results" / "eval" / f"{eval_id}-plots"
    if eval_config.num_plot_instances > 0:
        plots_dir.mkdir(parents=True, exist_ok=True)
        for plot_idx, snapshots in enumerate(result.plot_snapshots):
            fig = plot_lns_progress(snapshots, result.step_labels, epoch=final_epoch)
            fig.savefig(plots_dir / f"instance_{plot_idx:04d}.pdf", bbox_inches="tight")
            plt.close(fig)

    if train_config.logging.wandb_project:
        wandb.init(
            project=train_config.logging.wandb_project,
            name=f"eval-{train_config.run_name}-{eval_id}",
            job_type="eval",
            group=train_config.run_name,
            config={
                "eval": eval_config.model_dump(),
                "source_run": train_config.run_name,
                "source_run_dir": chosen_run_name,
                "source_epoch": final_epoch,
                "method": method,
                "train_config": train_config.model_dump(),
            },
        )
        upload_eval_results(
            run_name=train_config.run_name,
            eval_parquet_path=parquet_path,
            eval_id=eval_id,
            metadata={
                "eval_id": eval_id,
                "source_epoch": final_epoch,
                "method": method,
                "num_test_instances": eval_config.num_test_instances,
                "test_seed": eval_config.test_seed,
                "num_improvement_steps": eval_config.num_improvement_steps,
            },
        )
        if eval_config.num_plot_instances > 0:
            upload_eval_plots(run_name=train_config.run_name, eval_id=eval_id, plots_dir=plots_dir)
        wandb.finish()

    summary_md = (
        f"**Eval complete.** id=`{eval_id}` method=`{method}` "
        f"instances={eval_config.num_test_instances} steps={eval_config.num_improvement_steps}\n\n"
        f"- parquet: `{parquet_path}`\n"
        f"- plots: `{plots_dir}` ({eval_config.num_plot_instances} PDFs)\n"
    )
    mo.md(summary_md)
    return


if __name__ == "__main__":
    app.run()
