import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import os
    from copy import deepcopy

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from core.eval.config import EvaluationConfig
    from core.eval.problem_setting import SyntheticProblemSetting
    from core.evaluation import Evaluator
    from core.methods import build_method
    from core.problems import make_problem_config
    from figures.utils import project_root, save_figure

    # build_method resolves the method's run_dir relative to the working directory, so
    # normalise to the repo root regardless of where the notebook is launched from.
    os.chdir(project_root())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Rollouts vs steps

    The improvement step applies all K rollouts one after
    another, so the effective number of LNS steps is `I x K`, where `I` is the number of improvement steps.
    The baseline is `100 x 100 = 10,000` moves.

    This sweep trades rollouts for steps at a **fixed total move budget**: fewer rollouts
    (shorter steps) but more steps. Smaller `num_rollouts` re-queries the model more often
    (more adaptive destroy) and decays the SA temperature more finely, at the cost of more
    model forwards. Each variant is run on the first `NUM_INSTANCES` CVRP100 test instances;
    the two figures overlay one mean-objective descent curve per variant against wall-clock
    time and against effective LNS moves.
    """)
    return


@app.cell
def _():
    CONFIG_PATH = project_root() / "experiments/configs/eval/cvrp/nds_cvrp_100_split+simple.yml"
    DATASET_PATH = project_root() / "data/synthetic/cvrp/vrp100_test_seed1234.pkl"

    BUDGET = 10_000
    # (num_rollouts, num_improvement_steps), each holding num_rollouts * steps == BUDGET.
    VARIANTS = [
        (100, 100),
        # (50, 200),
        # (20, 500),
        (10, 1000),
        # (5, 2000),
        (1, 10000),
    ]
    for _rollouts, _steps in VARIANTS:
        assert _rollouts * _steps == BUDGET, (_rollouts, _steps)

    NUM_INSTANCES = 100
    BATCH_SIZE = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = project_root() / "figures" / "out" / "rollouts_vs_steps"
    return (
        BATCH_SIZE,
        CONFIG_PATH,
        DATASET_PATH,
        DEVICE,
        NUM_INSTANCES,
        OUT_DIR,
        VARIANTS,
    )


@app.cell
def _(BATCH_SIZE, CONFIG_PATH, DATASET_PATH, DEVICE, NUM_INSTANCES):
    eval_config = EvaluationConfig.from_file(CONFIG_PATH)
    problem = make_problem_config(name="cvrp", size=100)

    setting = SyntheticProblemSetting(
        problem=problem,
        dataset_path=str(DATASET_PATH),
        num_test_instances=NUM_INSTANCES,
        batch_size=BATCH_SIZE,
    )

    def method_for(num_rollouts: int, num_improvement_steps: int):
        config = deepcopy(eval_config.method)
        config.search.num_rollouts = num_rollouts
        config.search.num_improvement_steps = num_improvement_steps
        return build_method(config, problem=problem, device=DEVICE)

    return method_for, setting


@app.cell
def _():
    run = mo.ui.run_button(label="Run sweep")
    run
    return (run,)


@app.cell
def _(DEVICE, VARIANTS, method_for, run, setting):
    mo.stop(not run.value, mo.md("Click **Run sweep** to evaluate every variant."))

    # (num_rollouts, num_improvement_steps) -> per-step descent, wall-clock, effective moves.
    results: dict[tuple[int, int], dict[str, list[float]]] = {}
    with mo.status.progress_bar(
        total=len(VARIANTS),
        title="Rollouts-vs-steps sweep",
        subtitle=f"0/{len(VARIANTS)} variants",
    ) as variants_bar:
        for done, (num_rollouts, num_steps) in enumerate(VARIANTS):
            variants_bar.update(subtitle=f"R={num_rollouts}, S={num_steps}")
            method = method_for(num_rollouts, num_steps)
            evaluator = Evaluator(
                device=DEVICE,
                batch_progress=lambda it: mo.status.progress_bar(
                    it, title="Batches", remove_on_exit=True
                ),
                step_progress=lambda it: mo.status.progress_bar(
                    it, title="Improvement steps", remove_on_exit=True
                ),
            )
            result = evaluator.evaluate(setting, method)
            results[(num_rollouts, num_steps)] = {
                "obj": result.objective_per_step.mean(dim=1).tolist(),
                "time": list(result.time_per_step),
                "moves": [step * num_rollouts for step in range(num_steps + 1)],
            }
            del method
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            variants_bar.update(subtitle=f"{done + 1}/{len(VARIANTS)} variants")
    return (results,)


@app.function
def plot_curves(
    results: dict[tuple[int, int], dict[str, list[float]]],
    x_key: str,
    x_label: str,
    title: str,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    for (num_rollouts, num_steps), curve in results.items():
        n = len(curve["obj"])
        # Cap the number of markers so the dense low-rollout curves stay readable.
        markevery = max(1, n // 15)
        marks = list(range(0, n, markevery))
        if n and (n - 1) not in marks:
            marks.append(n - 1)
        ax.plot(
            curve[x_key],
            curve["obj"],
            marker="o",
            markersize=3,
            markevery=marks,
            label=f"K={num_rollouts}, I={num_steps}",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Objective")
    ax.set_title("CVRP100")
    ax.set_xlim(left=0)
    ax.legend()
    fig.tight_layout()
    return fig


@app.cell
def _(results: dict[tuple[int, int], dict[str, list[float]]]):
    fig_time = plot_curves(results, "time", "Total time (s)", "Objective vs wall-clock time")
    fig_moves = plot_curves(
        results, "moves", "Effective LNS moves", "Objective vs effective LNS moves"
    )
    mo.vstack([fig_time, fig_moves])
    return fig_moves, fig_time


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save figures")
    save_button
    return (save_button,)


@app.cell
def _(OUT_DIR, fig_moves, fig_time, save_button):
    mo.stop(not save_button.value, mo.md("Click **Save figures** to write the PDFs."))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _saved = []
    for _stem, _fig in {"obj_time": fig_time, "obj_moves": fig_moves}.items():
        _path = OUT_DIR / f"{_stem}.pdf"
        save_figure(_fig, _path)
        _saved.append(_path)

    mo.md("Saved:\n" + "\n".join(f"- `{p}`" for p in _saved))
    return


if __name__ == "__main__":
    app.run()
