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
    from rich.console import Console
    from rich.table import Table
    from wigglystuff import EnvConfig

    import wandb
    from core.eval_config import EvaluationConfig
    from core.lns.config import SearchParams
    from core.methods import (
        RANDOM_DESTROY_PRESETS,
        NDSPretrainedConfig,
        NDSPretrainedMethod,
        NDSTrainedConfig,
        NDSTrainedMethod,
        RandomDestroyMethod,
    )
    from core.methods.base import Method
    from core.models.nds.config import TrainConfig
    from core.models.nds.evaluation import configure_evaluation_step, plot_lns_progress
    from core.problems.setup import get_functions_by_problem, make_sampler
    from core.results.wandb_artifacts import upload_eval_plots, upload_eval_results
    from core.results.writer import write_eval_results

    load_dotenv(".env")

    RUNS_DIR = Path(__file__).parent / "runs"
    EVAL_CONFIGS_DIR = Path(__file__).parent / "configs" / "eval"
    PRETRAINED_DIR = Path(__file__).parent / "pretrained" / "nds"
    BASELINE_RESULTS_DIR = Path(__file__).parent / "results" / "eval"

    METHOD_TYPES = ("nds-trained", "nds-pretrained", "random-destroy")


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


@app.cell(hide_code=True)
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return (device,)


@app.cell(hide_code=True)
def _():
    is_script_mode = mo.app_meta().mode == "script"

    env_config_widget = mo.ui.anywidget(
        EnvConfig(
            {
                "WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True),
            }
        )
    )

    cli_eval_yaml: dict = {}
    cli_method_type: str | None = None
    cli_run_name: str | None = None
    if is_script_mode:
        _cli_args = mo.cli_args()

        if not _cli_args or _cli_args.get("help"):
            _usage = Table(title="NDS Evaluation -- CLI Usage")
            _usage.add_column("Flag", style="cyan")
            _usage.add_column("Description")
            _usage.add_row("--method <type>", "One of: " + ", ".join(METHOD_TYPES))
            _usage.add_row("--run <run_name>", "Training run (for method=nds-trained).")
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

        cli_method_type = str(_cli_args.get("method")) if _cli_args.get("method") else None
        cli_run_name = str(_cli_args.get("run")) if _cli_args.get("run") else None
        _eval_config_path = _cli_args.get("eval-config") or _cli_args.get("eval_config")
        if _eval_config_path:
            with open(str(_eval_config_path)) as _f:
                cli_eval_yaml = yaml.safe_load(_f) or {}

        for _key, _value in _cli_args.items():
            if _key in ("method", "run", "eval-config", "eval_config", "help"):
                continue
            cli_eval_yaml[_key] = _value
    return cli_eval_yaml, cli_method_type, cli_run_name, env_config_widget, is_script_mode


@app.cell(hide_code=True)
def _(env_config_widget, is_script_mode):
    env_config_widget if not is_script_mode and not wandb.login() else None
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Method
    """)
    return


@app.cell(hide_code=True)
def _(cli_method_type, is_script_mode):
    if is_script_mode:
        method_type_dropdown = None
        method_type = cli_method_type or "nds-trained"
    else:
        method_type_dropdown = mo.ui.dropdown(
            list(METHOD_TYPES), value="nds-trained", label="Method type"
        )
        method_type = method_type_dropdown.value
    method_type_dropdown
    return method_type, method_type_dropdown


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
def _(cli_run_name, completed_runs, is_script_mode, method_type):
    needs_run = method_type == "nds-trained"
    if is_script_mode:
        run_dropdown = None
        chosen_run_name = cli_run_name if needs_run else None
        if needs_run and chosen_run_name is None:
            raise ValueError("Pass --run <run_name> when method=nds-trained.")
    elif needs_run:
        mo.stop(not completed_runs, mo.md("**No completed runs found under `experiments/runs/`.**"))
        run_dropdown = mo.ui.dropdown(
            completed_runs, value=completed_runs[-1], label="Training run"
        )
        chosen_run_name = run_dropdown.value
    else:
        run_dropdown = None
        chosen_run_name = None
    run_dropdown
    return chosen_run_name, run_dropdown


@app.cell(hide_code=True)
def _(chosen_run_name):
    if chosen_run_name is None:
        run_dir = None
        train_config = None
    else:
        run_dir = RUNS_DIR / chosen_run_name
        if not run_dir.exists():
            raise ValueError(f"Run dir not found: {run_dir}")
        train_config = TrainConfig.model_validate(
            yaml.safe_load((run_dir / "config.yml").read_text())
        )
    return run_dir, train_config


@app.cell(hide_code=True)
def _():
    _sf = SearchParams.model_fields
    search_num_improvement_steps_widget = mo.ui.number(
        1, 5000, value=200, step=10, label=_sf["num_improvement_steps"].description
    )
    search_num_rollouts_widget = mo.ui.number(
        1, 1000, value=100, step=10, label=_sf["num_rollouts"].description
    )
    search_deconstruction_size_widget = mo.ui.number(
        1, 50, value=15, step=1, label=_sf["deconstruction_size"].description
    )
    search_n_repairs_widget = mo.ui.number(
        1, 10, value=5, step=1, label=_sf["n_repairs"].description
    )
    search_type_widget = mo.ui.dropdown(
        ["sequential", "parallel"], value="sequential", label="Improvement step variant"
    )
    return (
        search_deconstruction_size_widget,
        search_n_repairs_widget,
        search_num_improvement_steps_widget,
        search_num_rollouts_widget,
        search_type_widget,
    )


@app.cell(hide_code=True)
def _(
    method_type,
    search_deconstruction_size_widget,
    search_n_repairs_widget,
    search_num_improvement_steps_widget,
    search_num_rollouts_widget,
    search_type_widget,
):
    _out = (
        mo.vstack(
            [
                mo.md("### Search"),
                search_num_improvement_steps_widget,
                search_num_rollouts_widget,
                search_deconstruction_size_widget,
                search_n_repairs_widget,
                search_type_widget,
            ]
        )
        if method_type in ("nds-trained", "nds-pretrained")
        else None
    )
    _out
    return


@app.cell(hide_code=True)
def _():
    pretrained_options = {"(none)": ""}
    if PRETRAINED_DIR.exists():
        for _pt in sorted(PRETRAINED_DIR.glob("**/*.pt")):
            pretrained_options[f"{_pt.parent.name}/{_pt.name}"] = str(_pt)
    pretrained_path_widget = mo.ui.dropdown(
        options=pretrained_options, value="(none)", label="Pretrained checkpoint"
    )
    pretrained_load_decoder_widget = mo.ui.checkbox(value=False, label="Load decoder weights")
    pretrained_use_hidden_widget = mo.ui.checkbox(
        value=True, label="Use hidden decoding state (AR decoder)"
    )
    return (
        pretrained_load_decoder_widget,
        pretrained_path_widget,
        pretrained_use_hidden_widget,
    )


@app.cell(hide_code=True)
def _(
    method_type,
    pretrained_load_decoder_widget,
    pretrained_path_widget,
    pretrained_use_hidden_widget,
):
    _out = (
        mo.vstack(
            [
                mo.md("### Pretrained checkpoint"),
                pretrained_path_widget,
                pretrained_load_decoder_widget,
                pretrained_use_hidden_widget,
            ]
        )
        if method_type == "nds-pretrained"
        else None
    )
    _out
    return


@app.cell(hide_code=True)
def _():
    random_destroy_preset_widget = mo.ui.dropdown(
        list(RANDOM_DESTROY_PRESETS.keys()),
        value=next(iter(RANDOM_DESTROY_PRESETS)),
        label="Random-destroy preset",
    )
    return (random_destroy_preset_widget,)


@app.cell(hide_code=True)
def _(method_type, random_destroy_preset_widget):
    _out = (
        mo.vstack([mo.md("### Random destroy"), random_destroy_preset_widget])
        if method_type == "random-destroy"
        else None
    )
    _out
    return


@app.cell(hide_code=True)
def _(method_type, name_override_widget):
    _out = name_override_widget if method_type else None
    _out
    return


@app.cell(hide_code=True)
def _():
    name_override_widget = mo.ui.text(value="", label="Method name override (optional)")
    return (name_override_widget,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation config
    """)
    return


@app.cell(hide_code=True)
def _():
    presets = (
        ["(defaults)"] + sorted(p.name for p in EVAL_CONFIGS_DIR.glob("*.y*ml"))
        if EVAL_CONFIGS_DIR.exists()
        else ["(defaults)"]
    )
    return (presets,)


@app.cell(hide_code=True)
def _(is_script_mode, presets):
    if is_script_mode:
        preset_dropdown = None
    else:
        preset_dropdown = mo.ui.dropdown(presets, value=presets[0], label="Evaluation preset")
    preset_dropdown
    return (preset_dropdown,)


@app.cell(hide_code=True)
def _(is_script_mode, preset_dropdown):
    preset_dropdown_value = "(defaults)" if is_script_mode else preset_dropdown.value
    return (preset_dropdown_value,)


@app.cell(hide_code=True)
def _(cli_eval_yaml: dict, is_script_mode, preset_dropdown_value, train_config):
    if is_script_mode:
        preset_yaml = dict(cli_eval_yaml)
    elif preset_dropdown_value == "(defaults)":
        preset_yaml = {}
    else:
        with open(EVAL_CONFIGS_DIR / preset_dropdown_value) as _f:
            preset_yaml = yaml.safe_load(_f) or {}
    if train_config is not None:
        preset_yaml.setdefault("problem", train_config.problem.model_dump())
    preset_defaults = EvaluationConfig.model_validate(preset_yaml)
    return (preset_defaults,)


@app.cell(hide_code=True)
def _(preset_defaults):
    _ef = EvaluationConfig.model_fields

    num_test_instances_widget = mo.ui.number(
        1,
        10000,
        value=preset_defaults.num_test_instances,
        step=100,
        label=_ef["num_test_instances"].description,
    )
    test_seed_widget = mo.ui.number(
        0,
        2**31 - 1,
        value=preset_defaults.test_seed,
        step=1,
        label=_ef["test_seed"].description,
    )
    num_plot_instances_widget = mo.ui.number(
        0,
        100,
        value=preset_defaults.num_plot_instances,
        step=1,
        label=_ef["num_plot_instances"].description,
    )
    mo.vstack([num_test_instances_widget, test_seed_widget, num_plot_instances_widget])
    return (
        num_plot_instances_widget,
        num_test_instances_widget,
        test_seed_widget,
    )


@app.cell(hide_code=True)
def _(
    is_script_mode,
    num_plot_instances_widget,
    num_test_instances_widget,
    preset_defaults,
    test_seed_widget,
):
    if is_script_mode:
        eval_config = preset_defaults
    else:
        eval_config = preset_defaults.model_copy(
            update={
                "num_test_instances": num_test_instances_widget.value,
                "test_seed": test_seed_widget.value,
                "num_plot_instances": num_plot_instances_widget.value,
            }
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
    ## Evaluate
    """)
    return


@app.cell
def _(
    device,
    eval_config,
    is_script_mode,
    method_type,
    name_override_widget,
    pretrained_load_decoder_widget,
    pretrained_path_widget,
    pretrained_use_hidden_widget,
    random_destroy_preset_widget,
    run_dir,
    run_eval_button,
    search_deconstruction_size_widget,
    search_n_repairs_widget,
    search_num_improvement_steps_widget,
    search_num_rollouts_widget,
    search_type_widget,
):
    if not is_script_mode:
        mo.stop(not run_eval_button.value, mo.md("Click 'Run evaluation' to start."))

    test_sampler = make_sampler(
        eval_config.problem.name, eval_config.problem.size, seed=eval_config.test_seed
    )
    generate_start, repair_fn, _ = get_functions_by_problem(eval_config.problem.name, noop=False)

    _name_override = name_override_widget.value.strip() or None

    _search = SearchParams(
        num_improvement_steps=search_num_improvement_steps_widget.value,
        num_rollouts=search_num_rollouts_widget.value,
        deconstruction_size=search_deconstruction_size_widget.value,
        n_repairs=search_n_repairs_widget.value,
        improvement_step_type=search_type_widget.value,
    )

    if method_type == "nds-trained":
        if run_dir is None:
            raise ValueError("nds-trained method requires a selected run.")
        method_config = NDSTrainedConfig(
            run_dir=run_dir, search=_search, name_override=_name_override
        )
        method: Method = NDSTrainedMethod.build(
            method_config,
            repair=repair_fn,
            generate_start_solution=generate_start,
            device=device,
        )
    elif method_type == "nds-pretrained":
        _ckpt_path = pretrained_path_widget.value
        if not _ckpt_path:
            raise ValueError("Select a pretrained checkpoint.")
        from core.models.nds.config import ModelParams
        from core.models.nds.decoders.config import AutoregressiveDecoderConfig

        method_config = NDSPretrainedConfig(
            checkpoint_path=Path(_ckpt_path),
            load_decoder=pretrained_load_decoder_widget.value,
            model=ModelParams(
                decoder=AutoregressiveDecoderConfig(
                    use_hidden_decoding_state=pretrained_use_hidden_widget.value
                )
            ),
            search=_search,
            name_override=_name_override,
        )
        method = NDSPretrainedMethod.build(
            method_config,
            problem=eval_config.problem,
            repair=repair_fn,
            generate_start_solution=generate_start,
            device=device,
        )
    elif method_type == "random-destroy":
        _base = RANDOM_DESTROY_PRESETS[random_destroy_preset_widget.value]
        method_config = _base.model_copy(update={"name_override": _name_override})
        method = RandomDestroyMethod.build(
            method_config, repair=repair_fn, generate_start_solution=generate_start
        )
    else:
        raise ValueError(f"Unknown method type: {method_type}")

    eval_step = configure_evaluation_step(method=method, eval_config=eval_config)
    return eval_step, method, method_config, test_sampler


@app.cell
def _(eval_config, eval_step, test_sampler):
    instance = test_sampler(eval_config.num_test_instances)
    result = eval_step(instance)
    return (result,)


@app.cell
def _(
    chosen_run_name,
    eval_config,
    method,
    method_config,
    method_type,
    result,
    run_dir,
    train_config,
):
    eval_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    problem_tag = f"{eval_config.problem.name}{eval_config.problem.size}"

    rows = [
        {
            "method": method.name,
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

    if run_dir is not None:
        artifacts_root = run_dir
    else:
        artifacts_root = BASELINE_RESULTS_DIR / method.name
        artifacts_root.mkdir(parents=True, exist_ok=True)

    parquet_path = write_eval_results(artifacts_root, eval_run_id=eval_id, rows=rows)

    plots_dir = artifacts_root / "results" / "eval" / f"{eval_id}-plots"
    if eval_config.num_plot_instances > 0 and result.plot_snapshots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        _epoch_label = 0 if train_config is None else None
        for plot_idx, snapshots in enumerate(result.plot_snapshots):
            fig = plot_lns_progress(snapshots, result.step_labels, epoch=_epoch_label or 0)
            fig.savefig(plots_dir / f"instance_{plot_idx:04d}.pdf", bbox_inches="tight")
            plt.close(fig)

    if train_config is not None and train_config.logging.wandb_project:
        wandb.init(
            project=train_config.logging.wandb_project,
            name=f"eval-{train_config.run_name}-{eval_id}",
            job_type="eval",
            group=train_config.run_name,
            config={
                "eval": eval_config.model_dump(),
                "method_type": method_type,
                "method": method.name,
                "method_config": method_config.model_dump(mode="json"),
                "source_run": train_config.run_name,
                "source_run_dir": chosen_run_name,
                "train_config": train_config.model_dump(),
            },
        )
        upload_eval_results(
            run_name=train_config.run_name,
            eval_parquet_path=parquet_path,
            eval_id=eval_id,
            metadata={
                "eval_id": eval_id,
                "method": method.name,
                "method_type": method_type,
                "num_test_instances": eval_config.num_test_instances,
                "test_seed": eval_config.test_seed,
                "num_improvement_steps": method.search.num_improvement_steps,
            },
        )
        if eval_config.num_plot_instances > 0 and plots_dir.exists():
            upload_eval_plots(run_name=train_config.run_name, eval_id=eval_id, plots_dir=plots_dir)
        wandb.finish()

    summary_md = (
        f"**Eval complete.** id=`{eval_id}` method=`{method.name}` "
        f"instances={eval_config.num_test_instances} "
        f"steps={method.search.num_improvement_steps}\n\n"
        f"- parquet: `{parquet_path}`\n"
        f"- plots: `{plots_dir}` ({eval_config.num_plot_instances} PDFs)\n"
    )
    mo.md(summary_md)
    return


if __name__ == "__main__":
    app.run()
