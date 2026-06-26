import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import torch
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table as RichTable
    from tqdm import tqdm

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
    from core.models.nds.config import JointModelParams
    from core.models.nds.decoders.config import AutoregressiveDecoderConfig
    from core.problems.cvrp.dataset import load_cvrplib_instance, load_cvrplib_solution_bks
    from core.problems.solve import generate_initial_solution
    from core.solve import solve
    from core.ui.forms import build_mo_ui_dict, union_variants

    load_dotenv(".env")

    RUNS_DIR = Path(__file__).parent / "runs"
    PRETRAINED_DIR = Path(__file__).parent / "pretrained" / "nds"
    CVRPLIB_DIR = Path(__file__).parent.parent / "data" / "cvrplib"

    METHOD_TYPES = ("random-destroy", "nds-trained", "nds-pretrained")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CVRPLIB Benchmark Evaluation
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


@app.cell
def _(is_script_mode):
    cli_method_type: str | None = None
    cli_run_name: str | None = None
    cli_set: str | None = None
    cli_min_n: int = 0
    cli_max_n: int = 9999
    if is_script_mode:
        args = mo.cli_args()
        if not args or args.get("help"):
            usage = RichTable(title="CVRPLIB Evaluation -- CLI usage")
            usage.add_column("Flag", style="cyan")
            usage.add_column("Description")
            usage.add_row("--method <type>", "One of: " + ", ".join(METHOD_TYPES))
            usage.add_row("--run <run_name>", "Training run (for method=nds-trained).")
            usage.add_row("--set <name>", "CVRPLIB set name (e.g. X).")
            usage.add_row("--min-n <n>", "Min customers to evaluate (default 0).")
            usage.add_row("--max-n <n>", "Max customers to evaluate (default 9999).")
            Console().print(usage)
            sys.exit(0)
        cli_method_type = str(args.get("method")) if args.get("method") else None
        cli_run_name = str(args.get("run")) if args.get("run") else None
        cli_set = str(args.get("set")) if args.get("set") else None
        cli_min_n = int(args.get("min-n") or args.get("min_n") or 0)
        cli_max_n = int(args.get("max-n") or args.get("max_n") or 9999)
    return cli_max_n, cli_method_type, cli_min_n, cli_run_name, cli_set


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dataset
    """)
    return


@app.cell
def _():
    _available_sets = (
        [d.name for d in sorted(CVRPLIB_DIR.iterdir()) if d.is_dir()]
        if CVRPLIB_DIR.exists()
        else []
    )
    set_widget = mo.ui.dropdown(
        _available_sets,
        value=_available_sets[0] if _available_sets else None,
        label="CVRPLIB set",
    )
    min_n_widget = mo.ui.number(value=0, label="Min customers (0 = no filter)")
    max_n_widget = mo.ui.number(value=9999, label="Max customers (9999 = no filter)")
    mo.vstack([set_widget, min_n_widget, max_n_widget])
    return max_n_widget, min_n_widget, set_widget


@app.cell
def _(
    cli_max_n: int,
    cli_min_n: int,
    cli_set: str | None,
    is_script_mode,
    max_n_widget,
    min_n_widget,
    set_widget,
):
    chosen_set = cli_set if is_script_mode else set_widget.value
    chosen_min_n = cli_min_n if is_script_mode else int(min_n_widget.value)
    chosen_max_n = cli_max_n if is_script_mode else int(max_n_widget.value)
    return chosen_max_n, chosen_min_n, chosen_set


@app.cell
def _(chosen_max_n, chosen_min_n, chosen_set):
    def _n_customers_from_stem(stem: str) -> int:
        # e.g. "X-n101-k25" -> 100
        try:
            return int(stem.split("-n")[1].split("-")[0]) - 1
        except (IndexError, ValueError):
            return -1

    if chosen_set is not None:
        _set_dir = CVRPLIB_DIR / chosen_set
        vrp_paths = [
            p
            for p in sorted(_set_dir.glob("*.vrp"))
            if chosen_min_n <= _n_customers_from_stem(p.stem) <= chosen_max_n
        ]
    else:
        vrp_paths = []

    mo.md(f"**{len(vrp_paths)} instances** selected from set `{chosen_set}`.")
    return (vrp_paths,)


@app.cell
def _(vrp_paths):
    [Path(vrp).stem for vrp in vrp_paths]
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
    else:
        run_dir = RUNS_DIR / chosen_run_name
        if not run_dir.exists():
            raise ValueError(f"Run dir not found: {run_dir}")
    return (run_dir,)


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
    is_script_mode,
    method_type,
    pretrained_load_decoder_widget,
    pretrained_path_widget,
    pretrained_use_hidden_widget,
    random_destroy_seed_widget,
    run_dir,
    search_params,
):
    from core.problems.cvrp.config import CVRPConfig

    generate_start = generate_initial_solution
    _cvrp_problem = CVRPConfig()  # CVRPLIB is always CVRP

    if method_type == "random-destroy":
        method_config = RandomDestroyMethodConfig(
            search=search_params, seed=int(random_destroy_seed_widget.value)
        )
        method: Method = RandomDestroyMethod.build(
            method_config,
            problem=_cvrp_problem,
            generate_start_solution=generate_start,
        )
    elif method_type == "nds-trained":
        if run_dir is None:
            raise ValueError("nds-trained requires a selected run.")
        method_config = NDSTrainedConfig(run_dir=run_dir, search=search_params)
        method = NDSTrainedMethod.build(
            method_config, generate_start_solution=generate_start, device=device
        )
    elif method_type == "nds-pretrained":
        ckpt_path = pretrained_path_widget.value if not is_script_mode else ""
        if not ckpt_path:
            raise ValueError("Select a pretrained checkpoint.")
        method_config = NDSPretrainedConfig(
            checkpoint_path=Path(ckpt_path),
            load_decoder=pretrained_load_decoder_widget.value,
            model=JointModelParams(
                decoder=AutoregressiveDecoderConfig(
                    use_hidden_decoding_state=pretrained_use_hidden_widget.value
                )
            ),
            search=search_params,
        )
        method = NDSPretrainedMethod.build(
            method_config,
            problem=_cvrp_problem,
            generate_start_solution=generate_start,
            device=device,
        )
    else:
        raise ValueError(f"Unknown method type: {method_type}")

    run_eval_button = mo.ui.run_button(label="Run evaluation")
    run_eval_button if not is_script_mode else None
    return method, run_eval_button


@app.cell
def _(device, is_script_mode, method: Method, run_eval_button, vrp_paths):
    if not is_script_mode:
        mo.stop(not run_eval_button.value, mo.md("Click **Run evaluation** to start."))

    _iter = (
        tqdm(vrp_paths, desc="Instances")
        if is_script_mode
        else mo.status.progress_bar(vrp_paths, title="Instances", remove_on_exit=True)
    )

    _results = []
    for _vrp_path in _iter:
        _sol_path = _vrp_path.with_suffix(".sol")
        _instance = load_cvrplib_instance(_vrp_path).unsqueeze(0).to(device)
        _bks = load_cvrplib_solution_bks(_sol_path)
        _n_customers = _instance.num_customers

        with torch.no_grad():
            _final_solution = solve(method, _instance)

        _our_cost = int(_final_solution.total_distance(rounded=True).item())
        _gap_pct = (_our_cost - _bks) / _bks * 100

        _results.append(
            {
                "instance": _vrp_path.stem,
                "n_customers": _n_customers,
                "bks": _bks,
                "our_cost": _our_cost,
                "gap_pct": round(_gap_pct, 2),
            }
        )

    results_df = (
        pl.DataFrame(_results)
        if _results
        else pl.DataFrame(
            schema={
                "instance": pl.Utf8,
                "n_customers": pl.Int32,
                "bks": pl.Int32,
                "our_cost": pl.Int32,
                "gap_pct": pl.Float64,
            }
        )
    )
    return (results_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Results
    """)
    return


@app.cell
def _(results_df):
    mo.ui.table(results_df, selection=None)
    return


@app.cell
def _(results_df):
    if results_df.is_empty():
        summary_df = pl.DataFrame()
    else:
        summary_df = results_df.select(
            [
                pl.col("gap_pct").mean().round(3).alias("mean_gap_pct"),
                pl.col("gap_pct").median().round(3).alias("median_gap_pct"),
                pl.len().alias("n_instances"),
            ]
        )
    mo.ui.table(summary_df, selection=None)
    return


@app.cell
def _(results_df):
    if results_df.is_empty():
        bucketed_df = pl.DataFrame()
    else:
        bucketed_df = (
            results_df.with_columns(((pl.col("n_customers") // 100) * 100).alias("size_bucket"))
            .group_by("size_bucket")
            .agg(
                [
                    pl.col("gap_pct").mean().round(3).alias("mean_gap_pct"),
                    pl.col("gap_pct").median().round(3).alias("median_gap_pct"),
                    pl.len().alias("count"),
                ]
            )
            .sort("size_bucket")
        )
    mo.ui.table(bucketed_df, selection=None)
    return


if __name__ == "__main__":
    app.run()
