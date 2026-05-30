import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NDS Inference
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from drawdata import ScatterWidget

    from core.lns.acceptance_criterion import GreedyAcceptance
    from core.lns.operators.repair.greedy_insert import CVRPGreedyInsertRepair
    from core.lns.operators.repair.greedy_reconnect import TSPGreedyReconnectRepair
    from core.models.nds.config import TrainConfig
    from core.models.nds.factory import build_model
    from core.models.nds.improvement_step import NDSImprovementStep
    from core.problems.cvrp.instance import CVRPInstance
    from core.problems.cvrp.solve import one_customer_per_route
    from core.problems.tsp.instance import TSPInstance
    from core.problems.tsp.solve import random_permutation_graph

    return (
        CVRPGreedyInsertRepair,
        CVRPInstance,
        Path,
        ScatterWidget,
        TSPGreedyReconnectRepair,
        TSPInstance,
        TrainConfig,
        build_model,
        GreedyAcceptance,
        NDSImprovementStep,
        mo,
        np,
        one_customer_per_route,
        pd,
        plt,
        random_permutation_graph,
        torch,
        yaml,
    )


@app.cell
def _(Path, mo):
    RUNS_DIR = Path("experiments/runs")
    run_configs_map = {path.parent.name: path for path in sorted(RUNS_DIR.glob("*/config.yml"))}
    run_selector = mo.ui.dropdown(options=list(run_configs_map.keys()), label="Run")
    mo.vstack([mo.md("## 1. Select run"), run_selector])
    return run_configs_map, run_selector


@app.cell
def _(TrainConfig, mo, run_configs_map, run_selector, yaml):
    mo.stop(run_selector.value is None)
    config_path = run_configs_map[run_selector.value]
    config = TrainConfig.model_validate(yaml.safe_load(config_path.read_text()))
    mo.accordion(
        {"Config": mo.md(f"```yaml\n{yaml.dump(config.model_dump(), sort_keys=False)}\n```")}
    )
    return config, config_path


@app.cell
def _(config_path, mo):
    _ckpt_dir = config_path.parent / "checkpoints"
    checkpoint_paths = {int(p.stem.split("_")[1]): p for p in sorted(_ckpt_dir.glob("epoch_*.pt"))}
    epoch_selector = mo.ui.dropdown(
        options={str(e): e for e in sorted(checkpoint_paths)},
        label="Epoch",
    )
    load_button = mo.ui.button(label="Load model", value=False, on_click=lambda _: True)
    mo.vstack([mo.md("## 2. Load checkpoint"), mo.hstack([epoch_selector, load_button])])
    return checkpoint_paths, epoch_selector, load_button


@app.cell
def _(
    build_model,
    checkpoint_paths,
    config,
    epoch_selector,
    load_button,
    mo,
    torch,
):
    mo.stop(not load_button.value)
    mo.stop(epoch_selector.value is None)
    model = build_model(config)
    _ckpt = torch.load(
        checkpoint_paths[epoch_selector.value],
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(_ckpt["model"])
    model.eval()
    mo.callout(
        mo.md(
            f"Loaded epoch **{epoch_selector.value}**"
            f" - {config.problem.name.upper()}, {config.problem.size} nodes"
        ),
        kind="success",
    )
    return (model,)


@app.cell
def _(ScatterWidget, config, mo):
    _hint = (
        "**CVRP mode** - paint the **depot** with color **a** (first slot),"
        " customers with any other color."
        if config.problem.name == "cvrp"
        else "**TSP mode** - paint tour nodes with any color."
    )
    scatter_widget = mo.ui.anywidget(ScatterWidget(height=400))
    mo.vstack([mo.md(f"## 3. Define problem instance \n{_hint}"), scatter_widget])
    return (scatter_widget,)


@app.cell
def _(CVRPInstance, TSPInstance, config, mo, np, pd, scatter_widget, torch):
    _raw = scatter_widget.value.get("data", [])
    mo.stop(len(_raw) < 2)
    _df = pd.DataFrame(_raw)
    _xs = _df["x"].to_numpy(dtype=float)
    _ys = _df["y"].to_numpy(dtype=float)
    _scale = max(_xs.max() - _xs.min(), _ys.max() - _ys.min(), 1e-6)
    _xn = (_xs - _xs.min()) / _scale
    _yn = (_ys - _ys.min()) / _scale
    _coords = torch.tensor(np.stack([_xn, _yn], axis=1), dtype=torch.float32)

    if config.problem.name == "tsp":
        instance = TSPInstance.create(_coords.unsqueeze(0))
    else:
        _is_depot = _df["label"].to_numpy() == "a"
        mo.stop(
            _is_depot.sum() == 0,
            mo.callout(mo.md("Place at least one depot dot with color **a**."), kind="warn"),
        )
        mo.stop(
            (~_is_depot).sum() == 0,
            mo.callout(mo.md("Place at least one customer dot."), kind="warn"),
        )
        _locs = torch.cat([_coords[_is_depot][:1], _coords[~_is_depot]]).unsqueeze(0)
        _n_cust = int((~_is_depot).sum())
        _demand = torch.zeros(1, _n_cust + 1)
        _demand[0, 1:] = torch.rand(_n_cust) * 0.3 + 0.05
        instance = CVRPInstance.create(_locs, _demand, torch.ones(1, 1))
    return (instance,)


@app.cell
def _(config, instance, plt):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    instance[0].visualize(ax=_ax)
    _ax.set_title(f"{config.problem.name.upper()} - {instance.num_nodes} nodes")
    _fig
    return


@app.cell
def _(mo):
    num_rollouts_slider = mo.ui.slider(1, 16, value=4, label="Rollouts per step")
    num_steps_slider = mo.ui.slider(1, 100, value=30, label="LNS steps")
    mo.vstack(
        [mo.md("## 4. Inference parameters"), mo.hstack([num_rollouts_slider, num_steps_slider])]
    )
    return num_rollouts_slider, num_steps_slider


@app.cell
def _(mo):
    current_solution, set_current_solution = mo.state(None)
    objectives, set_objectives = mo.state([])
    return current_solution, objectives, set_current_solution, set_objectives


@app.cell(hide_code=True)
def _(
    CVRPGreedyInsertRepair,
    GreedyAcceptance,
    NDSImprovementStep,
    TSPGreedyReconnectRepair,
    config,
    current_solution,
    instance,
    mo,
    model,
    num_rollouts_slider,
    num_steps_slider,
    objectives,
    one_customer_per_route,
    random_permutation_graph,
    set_current_solution,
    set_objectives,
):
    def _make_initial():
        if config.problem.name == "tsp":
            return random_permutation_graph(instance)
        return one_customer_per_route(instance)

    def _make_step_fn():
        repair = (
            TSPGreedyReconnectRepair() if config.problem.name == "tsp" else CVRPGreedyInsertRepair()
        )
        step = NDSImprovementStep(
            model=model,
            repair=repair,
            num_rollouts=num_rollouts_slider.value,
            deconstruction_size=config.trainer.deconstruction_size,
            n_repairs=1,
        )
        return lambda sol: step(sol, GreedyAcceptance())

    def _on_reset(_):
        sol = _make_initial()
        set_current_solution(sol)
        set_objectives([sol.objective().mean().item()])

    def _on_step(count):
        sol = current_solution()
        if sol is None:
            sol = _make_initial()
            new_objs = [sol.objective().mean().item()]
        else:
            sol = _make_step_fn()(sol)
            new_objs = [*objectives(), sol.objective().mean().item()]
        set_current_solution(sol)
        set_objectives(new_objs)
        return count + 1

    def _on_run_all(count):
        sol = current_solution()
        if sol is None:
            sol = _make_initial()
        current_objs = list(objectives()) if objectives() else [sol.objective().mean().item()]
        step_fn = _make_step_fn()
        remaining = num_steps_slider.value - (len(current_objs) - 1)
        for _ in range(max(0, remaining)):
            sol = step_fn(sol)
            current_objs.append(sol.objective().mean().item())
        set_current_solution(sol)
        set_objectives(current_objs)
        return count + 1

    reset_button = mo.ui.button(label="Reset ↺", on_click=_on_reset)
    step_button = mo.ui.button(label="Step ▶", value=0, on_click=_on_step)
    run_button = mo.ui.button(label="Run All ▶▶", value=0, on_click=_on_run_all)
    mo.vstack([mo.md("## 5. Run LNS"), mo.hstack([reset_button, step_button, run_button])])
    return


@app.cell
def _(current_solution, mo, objectives, plt):
    sol = current_solution()
    mo.stop(sol is None, mo.md("Press **Reset ↺** or **Step ▶** to start."))

    _objs = objectives()
    _fig, (_ax_sol, _ax_curve) = plt.subplots(1, 2, figsize=(12, 5))

    sol[0].visualize(ax=_ax_sol)
    _ax_sol.set_title(f"Objective: {_objs[-1]:.4f}")

    _ax_curve.plot(_objs, marker="o", markersize=3)
    _ax_curve.set_xlabel("Step")
    _ax_curve.set_ylabel("Objective")
    _ax_curve.grid(True, alpha=0.3)
    if len(_objs) > 1:
        _improvement = 100 * (_objs[0] - _objs[-1]) / _objs[0]
        _ax_curve.set_title(
            f"LNS progress  ({_objs[0]:.4f} → {_objs[-1]:.4f}, {_improvement:.1f}% improvement)"
        )
    else:
        _ax_curve.set_title("LNS progress")

    _fig
    return


if __name__ == "__main__":
    app.run()
