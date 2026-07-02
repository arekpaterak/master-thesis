import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import torch
    import yaml
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table as RichTable
    from tqdm import tqdm
    from wigglystuff import EnvConfig

    import wandb
    from core.eval_config import EvaluationConfig
    from core.evaluation import Evaluator, PlotSnapshotCallback
    from core.lns.acceptance_criterion import AcceptanceCriterionConfig
    from core.lns.config import SearchParams
    from core.methods import (
        NDSPretrainedConfig,
        NDSPretrainedMethod,
        NDSTrainedConfig,
        NDSTrainedMethod,
        RandomDestroyMethod,
        RandomDestroyMethodConfig,
    )
    from core.methods.base import Method
    from core.models.nds.config import JointModelParams, TrainConfig
    from core.models.nds.decoders.config import AutoregressiveDecoderConfig
    from core.models.nds.encoders.config import EncoderConfig
    from core.models.nds.pretrained.config import (
        ORIGINAL_NDS_ENCODING_ORDER,
        ORIGINAL_NDS_ROUTE_AGGREGATION,
    )
    from core.problems import make_problem_config
    from core.problems.solve import generate_initial_solution
    from core.results.plots import plot_search_progress
    from core.results.wandb_artifacts import upload_eval_plots, upload_eval_results
    from core.results.writer import write_eval_results
    from core.ui.forms import build_mo_ui_dict, union_variants

    load_dotenv(".env")

    RUNS_DIR = Path(__file__).parent / "runs"
    EVAL_CONFIGS_DIR = Path(__file__).parent / "configs" / "eval"
    PRETRAINED_DIR = Path(__file__).parent / "pretrained" / "nds"

    METHOD_TYPES = ("random-destroy", "nds-trained", "nds-pretrained")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Evaluation
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
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_widget = mo.ui.dropdown(["cuda", "cpu"], value=_default_device, label="Device")
    device_widget
    return (device_widget,)


@app.cell
def _(device_widget):
    device = torch.device(device_widget.value)
    device
    return (device,)


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _(is_script_mode):
    cli_eval_yaml: dict = {}
    cli_method_type: str | None = None
    cli_run_name: str | None = None
    if is_script_mode:
        args = mo.cli_args()
        if not args or args.get("help"):
            usage = RichTable(title="Evaluation -- CLI usage")
            usage.add_column("Flag", style="cyan")
            usage.add_column("Description")
            usage.add_row("--method <type>", "One of: " + ", ".join(METHOD_TYPES))
            usage.add_row("--run <run_name>", "Training run (for method=nds-trained).")
            usage.add_row("--eval-config <path>", "YAML preset for EvaluationConfig.")
            Console().print(usage)
            sys.exit(0)
        cli_method_type = str(args.get("method")) if args.get("method") else None
        cli_run_name = str(args.get("run")) if args.get("run") else None
        eval_config_path = args.get("eval-config") or args.get("eval_config")
        if eval_config_path:
            with open(str(eval_config_path)) as _f:
                cli_eval_yaml = yaml.safe_load(_f) or {}
    return cli_eval_yaml, cli_method_type, cli_run_name


@app.cell(hide_code=True)
def _():
    env_config_widget = mo.ui.anywidget(
        EnvConfig({"WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True)})
    )
    return (env_config_widget,)


@app.cell(hide_code=True)
def _(env_config_widget, is_script_mode):
    env_config_widget if not is_script_mode and not wandb.login() else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method
    """)
    return


@app.cell(hide_code=True)
def _():
    method_type_widget = mo.ui.dropdown(
        list(METHOD_TYPES), value="random-destroy", label="Method type"
    )
    method_type_widget
    return (method_type_widget,)


@app.cell(hide_code=True)
def _(cli_method_type: str | None, is_script_mode, method_type_widget):
    method_type = cli_method_type if is_script_mode else method_type_widget.value
    return (method_type,)


@app.cell(hide_code=True)
def _():
    random_destroy_seed_widget = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Destroy seed")
    return (random_destroy_seed_widget,)


@app.cell(hide_code=True)
def _(method_type, random_destroy_seed_widget):
    mo.vstack(
        [mo.md("### Random destroy"), random_destroy_seed_widget]
    ) if method_type == "random-destroy" else None
    return


@app.cell(hide_code=True)
def _():
    completed_runs: list[str] = []
    if RUNS_DIR.exists():
        for candidate in sorted(RUNS_DIR.glob("nds-*")):
            if (
                candidate.is_dir()
                and (candidate / "config.yml").exists()
                and list((candidate / "checkpoints").glob("epoch_*.pt"))
            ):
                completed_runs.append(candidate.name)

    run_dropdown_widget = mo.ui.dropdown(
        completed_runs,
        value=completed_runs[-1] if completed_runs else None,
        label="Training run",
    )
    return (run_dropdown_widget,)


@app.cell(hide_code=True)
def _(method_type, run_dropdown_widget):
    run_dropdown_widget if method_type == "nds-trained" else None
    return


@app.cell(hide_code=True)
def _(
    cli_run_name: str | None,
    is_script_mode,
    method_type,
    run_dropdown_widget,
):
    if method_type != "nds-trained":
        chosen_run_name = None
    elif is_script_mode:
        if not cli_run_name:
            raise ValueError("Pass --run <run_name> when method=nds-trained.")
        chosen_run_name = cli_run_name
    else:
        chosen_run_name = run_dropdown_widget.value
    return (chosen_run_name,)


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
    _pretrained_options = {"(none)": ""}
    if PRETRAINED_DIR.exists():
        for ckpt in sorted(PRETRAINED_DIR.glob("**/*.pt")):
            _pretrained_options[f"{ckpt.parent.name}/{ckpt.name}"] = str(ckpt)
    pretrained_path_widget = mo.ui.dropdown(
        options=_pretrained_options, value="(none)", label="Pretrained checkpoint"
    )
    pretrained_load_decoder_widget = mo.ui.checkbox(value=False, label="Load decoder weights")
    pretrained_use_hidden_widget = mo.ui.checkbox(value=True, label="Use hidden decoding state")
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
    mo.vstack(
        [
            mo.md("### Pretrained NDS"),
            pretrained_path_widget,
            pretrained_load_decoder_widget,
            pretrained_use_hidden_widget,
        ]
    ) if method_type == "nds-pretrained" else None
    return


@app.cell(hide_code=True)
def _():
    search_defaults = SearchParams(
        repair_type="greedy_insert",
        num_improvement_steps=50,
        num_rollouts=100,
        deconstruction_size=15,
        n_repairs=5,
        improvement_step_type="numba_sequential",
        n_augmentations=1,
    )
    search_form = build_mo_ui_dict(SearchParams, search_defaults, skip={"acceptance"})
    return (search_form,)


@app.cell(hide_code=True)
def _():
    acceptance_variants = union_variants(AcceptanceCriterionConfig)
    acceptance_type_widget = mo.ui.dropdown(
        list(acceptance_variants), value="sa", label="acceptance"
    )
    return acceptance_type_widget, acceptance_variants


@app.cell(hide_code=True)
def _(acceptance_type_widget, acceptance_variants):
    acceptance_variant_cls = acceptance_variants[acceptance_type_widget.value]
    acceptance_form = build_mo_ui_dict(acceptance_variant_cls, acceptance_variant_cls())
    return acceptance_form, acceptance_variant_cls


@app.cell(hide_code=True)
def _(acceptance_form, acceptance_type_widget, search_form):
    mo.vstack(
        [
            mo.md("### Search"),
            *search_form.values(),
            acceptance_type_widget,
            *acceptance_form.values(),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation config
    """)
    return


@app.cell
def _():
    presets = ["(defaults)"]
    if EVAL_CONFIGS_DIR.exists():
        presets += sorted(p.name for p in EVAL_CONFIGS_DIR.glob("*.y*ml"))
    preset_dropdown_widget = mo.ui.dropdown(presets, value=presets[0], label="Evaluation preset")
    preset_dropdown_widget
    return (preset_dropdown_widget,)


@app.cell
def _(cli_eval_yaml: dict, is_script_mode, preset_dropdown_widget):
    if is_script_mode:
        preset_yaml = dict(cli_eval_yaml)
    elif preset_dropdown_widget.value == "(defaults)":
        preset_yaml = {}
    else:
        with open(EVAL_CONFIGS_DIR / preset_dropdown_widget.value) as _f:
            preset_yaml = yaml.safe_load(_f) or {}
    preset_defaults = EvaluationConfig.model_validate(preset_yaml)
    return (preset_defaults,)


@app.cell(hide_code=True)
def _(preset_defaults):
    problem_name_widget = mo.ui.dropdown(
        ["cvrp", "tsp"], value=preset_defaults.problem.name, label="problem.name"
    )
    _sizes = [20, 50, 100]
    problem_size_widget = mo.ui.dropdown(
        _sizes,
        value=preset_defaults.problem.size if preset_defaults.problem.size in _sizes else 100,
        label="problem.size",
    )
    wandb_project_widget = mo.ui.text(
        value=preset_defaults.wandb_project, label="wandb_project (empty = no upload)"
    )
    num_test_instances_widget = mo.ui.number(
        0, 10000, value=preset_defaults.num_test_instances, step=100, label="num_test_instances"
    )
    batch_size_widget = mo.ui.number(value=preset_defaults.batch_size, label="batch_size")
    test_seed_widget = mo.ui.number(0, value=preset_defaults.test_seed, step=1, label="test_seed")
    _initial_path = (
        str(Path(preset_defaults.dataset_path).parent)
        if preset_defaults.dataset_path
        else str(Path(__file__).parent.parent / "data")
    )
    dataset_path_widget = mo.ui.file_browser(
        initial_path=_initial_path,
        multiple=False,
        label="dataset_path (select nothing = random sampling)",
    )
    num_plot_instances_widget = mo.ui.number(
        0, 100, value=preset_defaults.num_plot_instances, step=1, label="num_plot_instances"
    )
    mo.vstack(
        [
            problem_name_widget,
            problem_size_widget,
            num_test_instances_widget,
            batch_size_widget,
            test_seed_widget,
            dataset_path_widget,
            num_plot_instances_widget,
            wandb_project_widget,
        ]
    )
    return (
        batch_size_widget,
        dataset_path_widget,
        num_plot_instances_widget,
        num_test_instances_widget,
        problem_name_widget,
        problem_size_widget,
        test_seed_widget,
        wandb_project_widget,
    )


@app.cell
def _(
    batch_size_widget,
    dataset_path_widget,
    is_script_mode,
    num_plot_instances_widget,
    num_test_instances_widget,
    preset_defaults,
    problem_name_widget,
    problem_size_widget,
    test_seed_widget,
    wandb_project_widget,
):
    if is_script_mode:
        eval_config = preset_defaults
    else:
        eval_config = preset_defaults.model_copy(
            update={
                "problem": make_problem_config(
                    name=problem_name_widget.value, size=int(problem_size_widget.value)
                ),
                "num_test_instances": num_test_instances_widget.value,
                "batch_size": batch_size_widget.value,
                "test_seed": test_seed_widget.value,
                "dataset_path": dataset_path_widget.value[0].path
                if dataset_path_widget.value
                else None,
                "num_plot_instances": num_plot_instances_widget.value,
                "wandb_project": wandb_project_widget.value,
            }
        )
    return (eval_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluate
    """)
    return


@app.cell
def _(acceptance_form, acceptance_variant_cls, search_form):
    acceptance = acceptance_variant_cls(**acceptance_form.value)
    search_params = SearchParams.model_validate({**search_form.value, "acceptance": acceptance})
    return (search_params,)


@app.cell
def _(
    device,
    eval_config,
    is_script_mode,
    method_type,
    pretrained_load_decoder_widget,
    pretrained_path_widget,
    pretrained_use_hidden_widget,
    random_destroy_seed_widget,
    run_dir,
    search_params,
):
    generate_start = generate_initial_solution

    if method_type == "random-destroy":
        method_config = RandomDestroyMethodConfig(
            search=search_params, seed=int(random_destroy_seed_widget.value)
        )
        method: Method = RandomDestroyMethod.build(
            method_config, problem=eval_config.problem, generate_start_solution=generate_start
        )
    elif method_type == "nds-trained":
        if run_dir is None:
            raise ValueError("nds-trained requires a selected run.")
        method_config = NDSTrainedConfig(run_dir=run_dir, search=search_params)
        method = NDSTrainedMethod.build(
            method_config, generate_start_solution=generate_start, device=device
        )
    elif method_type == "nds-pretrained":
        ckpt_path = pretrained_path_widget.value
        if not ckpt_path:
            raise ValueError("Select a pretrained checkpoint.")
        method_config = NDSPretrainedConfig(
            checkpoint_path=Path(ckpt_path),
            load_decoder=pretrained_load_decoder_widget.value,
            model=JointModelParams(
                encoder=EncoderConfig(
                    encoding_order=ORIGINAL_NDS_ENCODING_ORDER,
                    route_aggregation=ORIGINAL_NDS_ROUTE_AGGREGATION,
                ),
                decoder=AutoregressiveDecoderConfig(
                    use_hidden_decoding_state=pretrained_use_hidden_widget.value
                ),
            ),
            search=search_params,
        )
        method = NDSPretrainedMethod.build(
            method_config,
            problem=eval_config.problem,
            generate_start_solution=generate_start,
            device=device,
        )
        print(f"Loaded pretrained encoder weights from {ckpt_path}")
        if pretrained_load_decoder_widget.value == True:
            print(f"Loaded pretrained decoder weights from {ckpt_path}")
        print(f"Using hidden state: {pretrained_use_hidden_widget.value}")
    else:
        raise ValueError(f"Unknown method type: {method_type}")

    run_eval_button = mo.ui.run_button(label="Run evaluation")
    run_eval_button if not is_script_mode else None
    return method, method_config, run_eval_button


@app.cell
def _(device, eval_config, is_script_mode, method: Method, run_eval_button):
    if not is_script_mode:
        mo.stop(not run_eval_button.value, mo.md("Click **Run evaluation** to start."))
    plot_snapshot_cb = PlotSnapshotCallback(num_plot_instances=eval_config.num_plot_instances)
    if is_script_mode:
        batch_progress = lambda it: tqdm(it, desc="Batches")
        step_progress = lambda it: tqdm(it, desc="Improvement steps", leave=False)
    else:
        batch_progress = lambda it: mo.status.progress_bar(it, title="Batches", remove_on_exit=True)
        step_progress = lambda it: mo.status.progress_bar(
            it, title="Improvement steps", remove_on_exit=True
        )
    evaluator = Evaluator(
        on_step=[plot_snapshot_cb.on_step],
        batch_progress=batch_progress,
        step_progress=step_progress,
        device=device,
    )
    result = evaluator.evaluate(eval_config, method)

    # Free GPU memory
    _model = getattr(method.improvement_step, "_model", None)
    if _model is not None:
        _model.to("cpu")
        torch.cuda.empty_cache()

    print("Finished evaluation!")
    return plot_snapshot_cb, result


@app.cell
def _():
    eval_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (eval_id,)


@app.cell
def _(eval_config):
    problem_tag = f"{eval_config.problem.name}{eval_config.problem.size}"
    return (problem_tag,)


@app.cell
def _(method: Method, run_dir):
    artifacts_root = run_dir if run_dir is not None else RUNS_DIR / method.name
    artifacts_root.mkdir(parents=True, exist_ok=True)
    return (artifacts_root,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Aggregated results
    """)
    return


@app.cell
def _(result):
    _obj = result.objective_per_step
    summary_df = pl.DataFrame(
        {
            "method": [result.method_name],
            "n_instances": [_obj.shape[1]],
            "n_steps": [_obj.shape[0] - 1],
            "obj_initial_mean": [_obj[0].mean().item()],
            "obj_final_mean": [_obj[-1].mean().item()],
            "improvement_mean": [(_obj[0] - _obj[-1]).mean().item()],
            "improvement_std": [(_obj[0] - _obj[-1]).std().item()],
        }
    )
    return (summary_df,)


@app.cell
def _(summary_df):
    mo.ui.table(summary_df, selection=None)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Results per instance
    """)
    return


@app.cell
def _(result):
    _po = result.objective_per_step
    per_instance_df = pl.DataFrame(
        {
            "instance": list(range(_po.shape[1])),
            "obj_initial": _po[0].tolist(),
            "obj_final": _po[-1].tolist(),
            "improvement": (_po[0] - _po[-1]).tolist(),
        }
    )
    return (per_instance_df,)


@app.cell
def _(per_instance_df):
    mo.ui.table(per_instance_df, selection=None)
    return


@app.cell
def _(result):
    result.objective_per_step.mean(dim=1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Search progress
    """)
    return


@app.cell
def _(result):
    _po = result.objective_per_step
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(result.step_labels, _po.mean(dim=1).tolist(), label="mean")
    _ax.fill_between(
        result.step_labels,
        (_po.mean(dim=1) - _po.std(dim=1)).tolist(),
        (_po.mean(dim=1) + _po.std(dim=1)).tolist(),
        alpha=0.2,
        label="+/- 1 std",
    )
    _ax.set_xlabel("Improvement step")
    _ax.set_ylabel("Objective")
    _ax.set_title(f"{result.method_name}: objective vs step")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(disabled=True)
def _(artifacts_root, eval_config, eval_id, problem_tag, result):
    rows = [
        {
            "method": result.method_name,
            "instance_id": f"{problem_tag}-seed{eval_config.test_seed}-{i:04d}",
            "seed": eval_config.test_seed,
            "lns_step": step,
            "objective": float(result.objective_per_step[step_idx, i].item()),
            "time_s": 0.0,
            "initial_objective": float(result.objective_per_step[0, i].item()),
        }
        for step_idx, step in enumerate(result.step_labels)
        for i in range(result.objective_per_step.shape[1])
    ]
    parquet_path = write_eval_results(artifacts_root, eval_run_id=eval_id, rows=rows)
    return (parquet_path,)


@app.cell(disabled=True)
def _(artifacts_root, eval_id, plot_snapshot_cb, result):
    plots_dir = artifacts_root / "results" / "eval" / f"{eval_id}-plots"
    if plot_snapshot_cb.snapshots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        for plot_idx, snapshots in enumerate(plot_snapshot_cb.snapshots):
            _fig = plot_search_progress(snapshots, result.step_labels, epoch=0)
            _fig.savefig(plots_dir / f"instance_{plot_idx:04d}.pdf", bbox_inches="tight")
            plt.close(_fig)
    return (plots_dir,)


@app.cell(disabled=True)
def _(
    chosen_run_name,
    eval_config,
    eval_id,
    method: Method,
    method_config,
    method_type,
    parquet_path,
    plot_snapshot_cb,
    plots_dir,
    train_config,
):
    if train_config is not None and train_config.logging.wandb_project:
        wandb_project = train_config.logging.wandb_project
        wandb_run_name = f"eval-{train_config.run_name}-{eval_id}"
        wandb_group = train_config.run_name
        artifact_run_name = train_config.run_name
        link_checkpoint_alias: str | None = "final"
        extra_config = {
            "source_run": train_config.run_name,
            "source_run_dir": chosen_run_name,
            "train_config": train_config.model_dump(),
        }
    elif eval_config.wandb_project:
        wandb_project = eval_config.wandb_project
        wandb_run_name = f"eval-{method.name}-{eval_id}"
        wandb_group = method.name
        artifact_run_name = method.name
        link_checkpoint_alias = None
        extra_config = {}
    else:
        wandb_project = None

    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            job_type="eval",
            group=wandb_group,
            config={
                "eval": eval_config.model_dump(),
                "method_type": method_type,
                "method": method.name,
                "method_config": method_config.model_dump(mode="json"),
                **extra_config,
            },
        )
        upload_eval_results(
            run_name=artifact_run_name,
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
            link_checkpoint_alias=link_checkpoint_alias,
        )
        if plot_snapshot_cb.snapshots and plots_dir.exists():
            upload_eval_plots(run_name=artifact_run_name, eval_id=eval_id, plots_dir=plots_dir)
        wandb.finish()
    return


if __name__ == "__main__":
    app.run()
