import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    from contextlib import contextmanager
    from time import perf_counter

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import yaml

    from core.lns.acceptance_criterion import build_criterion
    from core.lns.improvement_step import build_random_node_destroy_step
    from core.models.nds.config import TrainConfig
    from core.models.nds.improvement_step import build_nds_improvement_step
    from core.problems import registry
    from core.problems.registry import build_nds_model
    from figures.utils import (
        figsize,
        project_root,
        robust_mean,
        save_figure,
        set_thesis_style,
        set_white_background,
    )

    set_thesis_style()
    set_white_background()

    SEED = 0
    # Each method is measured into encode/search (one-time instance encoder), encode/iteration
    # and decode/iteration (summed over the search), and other (total minus encode and decode:
    # repair, acceptance, overhead). The plot combines both encode parts into one "encode"
    # segment; the exact values, including the instance encoder, are printed to the console.
    PLOT_SEGMENTS = ["encode", "decode", "other"]
    PLOT_COLORS = ["tab:blue", "tab:orange", "tab:gray"]
    RANDOM_LABEL = "Random"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Inference cost breakdown

    Per method, the wall-clock cost of one full LNS search as three stacked parts that sum to
    the total search time: **encode** (the one-time instance encode plus the per-iteration
    encode, summed), **decode** (per iteration, summed), and **other** -- the total minus
    encode and decode (repair, acceptance, overhead -- everything the model timers do not
    cover). Measured by instrumenting the model's own calls (`encode_instance`, the encoder
    forward, `decoder.rollout`) while running a real search inside `search_session`, so the
    Split encoder's one-time encode genuinely amortizes. The exact split, including the
    one-time instance encoder hidden inside the merged encode bar, is printed to the console.

    Every bar is therefore the same quantity (total search time), so methods are directly
    comparable. Random Destroy has no model, so its whole bar is other.

    Each method is timed over `N_BATCHES` searches on fresh instances and each component is
    the outlier-robust (IQR) mean over them. Pick problem, size, batch size, step count and
    rollout size, then click **Measure** (needs a GPU for representative numbers). Each timer
    brackets its call with `torch.cuda.synchronize()`.
    """)
    return


@app.cell
def _():
    # Problem title -> method label -> {problem size: train config path (under the repo root)}.
    TRAIN_DIR = "experiments/configs/train"
    CONFIGS: dict[str, dict[str, dict[int, str]]] = {
        "TSP": {
            "Joint + AR": {
                50: f"{TRAIN_DIR}/tsp/nds_tsp_50+ar.yml",
                100: f"{TRAIN_DIR}/tsp/nds_tsp_100+ar.yml",
            },
            "Joint + NAR": {
                50: f"{TRAIN_DIR}/tsp/nds_tsp_50+simple.yml",
                100: f"{TRAIN_DIR}/tsp/nds_tsp_100+simple.yml",
            },
            "Split + AR": {
                50: f"{TRAIN_DIR}/tsp/nds_tsp_50_split+ar.yml",
                100: f"{TRAIN_DIR}/tsp/nds_tsp_100_split+ar.yml",
            },
            "Split + NAR": {
                50: f"{TRAIN_DIR}/tsp/nds_tsp_50_split+simple.yml",
                100: f"{TRAIN_DIR}/tsp/nds_tsp_100_split+simple.yml",
            },
        },
        "CVRP": {
            "Joint + AR": {
                50: f"{TRAIN_DIR}/cvrp/nds_cvrp_50+ar.yml",
                100: f"{TRAIN_DIR}/cvrp/nds_cvrp_100+ar.yml",
            },
            "Joint + NAR": {
                50: f"{TRAIN_DIR}/cvrp/nds_cvrp_50+simple.yml",
                100: f"{TRAIN_DIR}/cvrp/nds_cvrp_100+simple.yml",
            },
            "Split + AR": {
                50: f"{TRAIN_DIR}/cvrp/nds_cvrp_50_split+ar.yml",
                100: f"{TRAIN_DIR}/cvrp/nds_cvrp_100_split+ar.yml",
            },
            "Split + NAR": {
                50: f"{TRAIN_DIR}/cvrp/nds_cvrp_50_split+simple.yml",
                100: f"{TRAIN_DIR}/cvrp/nds_cvrp_100_split+simple.yml",
            },
        },
        "CMSTP": {
            "Joint + AR": {
                50: f"{TRAIN_DIR}/cmstp/nds_cmstp_50+ar.yml",
                100: f"{TRAIN_DIR}/cmstp/nds_cmstp_100+ar.yml",
            },
            "Joint + NAR": {
                50: f"{TRAIN_DIR}/cmstp/nds_cmstp_50+simple.yml",
                100: f"{TRAIN_DIR}/cmstp/nds_cmstp_100+simple.yml",
            },
            "Split + AR": {
                50: f"{TRAIN_DIR}/cmstp/nds_cmstp_50_split+ar.yml",
                100: f"{TRAIN_DIR}/cmstp/nds_cmstp_100_split+ar.yml",
            },
            "Split + NAR": {
                50: f"{TRAIN_DIR}/cmstp/nds_cmstp_50_split+simple.yml",
                100: f"{TRAIN_DIR}/cmstp/nds_cmstp_100_split+simple.yml",
            },
        },
    }

    WARMUP_STEPS = 10
    N_BATCHES = 5  # timed searches per method; components are an outlier-robust mean over them
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT_DIR = project_root() / "figures" / "out"
    return CONFIGS, DEVICE, N_BATCHES, OUT_DIR, WARMUP_STEPS


@app.cell
def _(CONFIGS: dict[str, dict[str, dict[int, str]]]):
    sizes = sorted(
        {
            size
            for methods in CONFIGS.values()
            for config_by_size in methods.values()
            for size in config_by_size
        }
    )
    problem_ui = mo.ui.dropdown(options=list(CONFIGS), value="CVRP", label="Problem")
    size_ui = mo.ui.dropdown(
        options={str(size): size for size in sizes},
        value="100" if 100 in sizes else str(sizes[-1]),
        label="Problem size",
    )
    batch_size_ui = mo.ui.number(start=1, step=1, value=100, label="Batch size")
    num_steps_ui = mo.ui.number(start=1, step=1, value=100, label="Search steps")
    rollout_size_ui = mo.ui.number(start=1, step=1, value=100, label="Rollout size")
    measure_button = mo.ui.run_button(label="Measure")
    mo.vstack(
        [problem_ui, size_ui, batch_size_ui, num_steps_ui, rollout_size_ui, measure_button],
        justify="start",
        gap=1,
    )
    return (
        batch_size_ui,
        measure_button,
        num_steps_ui,
        problem_ui,
        rollout_size_ui,
        size_ui,
    )


@app.function
def load_config(config_path: str) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load((project_root() / config_path).read_text()))


@app.cell
def _(
    CONFIGS: dict[str, dict[str, dict[int, str]]],
    DEVICE,
    N_BATCHES,
    WARMUP_STEPS,
    batch_size_ui,
    measure_button,
    num_steps_ui,
    problem_ui,
    rollout_size_ui,
    size_ui,
):
    mo.stop(
        not measure_button.value,
        mo.md("Pick the settings and click **Measure** to time the searches."),
    )

    def sync() -> None:
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def component_timing(model):
        """Time the model's encode/search, per-iteration encode, and decode calls in place.

        Wraps the model's own methods and the encoder forward (via hooks); restores
        everything on exit. Each timer brackets the call with a CUDA sync.
        """
        buckets = {"encode/search": 0.0, "encode/iteration": 0.0, "decode/iteration": 0.0}
        starts: list[float] = []

        def pre_hook(_module, _inputs):
            sync()
            starts.append(perf_counter())

        def post_hook(_module, _inputs, _output):
            sync()
            buckets["encode/iteration"] += perf_counter() - starts.pop()

        pre_handle = model.encoder.register_forward_pre_hook(pre_hook)
        post_handle = model.encoder.register_forward_hook(post_hook)

        original_rollout = model.decoder.rollout

        def timed_rollout(*args, **kwargs):
            sync()
            start = perf_counter()
            result = original_rollout(*args, **kwargs)
            sync()
            buckets["decode/iteration"] += perf_counter() - start
            return result

        model.decoder.rollout = timed_rollout

        has_encode_instance = hasattr(type(model), "encode_instance")
        if has_encode_instance:
            original_encode = model.encode_instance

            def timed_encode(*args, **kwargs):
                sync()
                start = perf_counter()
                result = original_encode(*args, **kwargs)
                sync()
                buckets["encode/search"] += perf_counter() - start
                return result

            model.encode_instance = timed_encode

        try:
            yield buckets
        finally:
            pre_handle.remove()
            post_handle.remove()
            del model.decoder.rollout
            if has_encode_instance:
                del model.encode_instance

    def run_search(step, instance, num_steps, acceptance, initial_solution, progress=None):
        solution = initial_solution(instance)
        criterion = build_criterion(acceptance, n_steps=max(num_steps, 1))
        steps = range(num_steps)
        if progress is not None:
            steps = progress(steps)
        with step.search_session(solution):
            criterion.reset()
            for _ in steps:
                solution = step(solution, criterion)
                criterion.step()

    def build_step(config, rollout_size, model=None):
        search = config.search.model_copy(update={"num_rollouts": rollout_size})
        if model is None:
            return build_random_node_destroy_step(
                search, lns=registry.lns(config.problem.name), seed=SEED
            )
        return build_nds_improvement_step(
            search, model=model, lns=registry.lns(config.problem.name), seed=SEED
        )

    def timed_search(step, instance, num_steps, acceptance, initial_solution, model=None):
        """Run one search and return its total wall time plus (if a model) the component split."""
        sync()
        start = perf_counter()
        if model is None:
            run_search(step, instance, num_steps, acceptance, initial_solution)
            buckets = {"encode/search": 0.0, "encode/iteration": 0.0, "decode/iteration": 0.0}
        else:
            with component_timing(model) as buckets:
                run_search(step, instance, num_steps, acceptance, initial_solution)
        sync()
        total = perf_counter() - start
        # Whatever the model timers did not cover (repair, acceptance, etc.) is the remainder.
        return {**buckets, "other": max(total - sum(buckets.values()), 0.0)}

    def measure(config_path, num_steps, batch_size, rollout_size, with_model, progress=None):
        """Time N_BATCHES searches on fresh instances; return the robust mean per component."""
        config = load_config(config_path)
        sampler = registry.build_sampler(config.problem)
        initial_solution = registry.problem(config.problem.name).initial_solution
        model = build_nds_model(config.model).to(DEVICE).eval() if with_model else None
        step = build_step(config, rollout_size, model=model)

        run_search(
            step,
            sampler(batch_size).to(DEVICE),
            WARMUP_STEPS,
            config.search.acceptance,
            initial_solution,
        )

        keys = ["encode/search", "encode/iteration", "decode/iteration", "other"]
        samples: dict[str, list[float]] = {key: [] for key in keys}
        batches = range(N_BATCHES)
        if progress is not None:
            batches = progress(batches)
        for _ in batches:
            instance = sampler(batch_size).to(DEVICE)
            detail = timed_search(
                step, instance, num_steps, config.search.acceptance, initial_solution, model=model
            )
            for key in keys:
                samples[key].append(detail[key])

        del model, step
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return {key: robust_mean(values) for key, values in samples.items()}

    problem = problem_ui.value
    size = size_ui.value
    num_steps = num_steps_ui.value
    batch_size = batch_size_ui.value
    rollout_size = rollout_size_ui.value
    tasks = [
        (label, config_by_size[size])
        for label, config_by_size in CONFIGS[problem].items()
        if config_by_size.get(size)
    ]

    def report(label, detail):
        # Print exact values, including the one-time instance encoder hidden in the merged bar.
        print(
            f"{label}: instance_encode={detail['encode/search']:.4f}s "
            f"encode/iter={detail['encode/iteration']:.4f}s "
            f"decode/iter={detail['decode/iteration']:.4f}s "
            f"other={detail['other']:.4f}s total={sum(detail.values()):.4f}s"
        )

    means: dict[str, dict[str, float]] = {}
    with mo.status.progress_bar(
        total=len(tasks) + 1,
        title=f"Measuring {problem}{size}",
        subtitle=f"0/{len(tasks) + 1} methods",
    ) as methods_bar:
        for done, (label, config_path) in enumerate(tasks):
            means[label] = measure(
                config_path,
                num_steps,
                batch_size,
                rollout_size,
                with_model=True,
                progress=lambda it: mo.status.progress_bar(
                    it, title=label, subtitle="batches", remove_on_exit=True
                ),
            )
            report(label, means[label])
            methods_bar.update(subtitle=f"{done + 1}/{len(tasks) + 1} methods")
        # Random baseline: reuse the first task's config for its search/problem settings.
        means[RANDOM_LABEL] = measure(
            tasks[0][1], num_steps, batch_size, rollout_size, with_model=False
        )
        report(RANDOM_LABEL, means[RANDOM_LABEL])
        methods_bar.update(subtitle=f"{len(tasks) + 1}/{len(tasks) + 1} methods")
    return means, num_steps, problem


@app.function
def segment_value(detail: dict[str, float], segment: str) -> float:
    """Value of one plot segment, combining the two encode parts into a single "encode"."""
    if segment == "encode":
        return detail["encode/search"] + detail["encode/iteration"]
    if segment == "decode":
        return detail["decode/iteration"]
    return detail["other"]


@app.function
def plot_breakdown(means: dict[str, dict[str, float]], title: str, num_steps: int):
    """Horizontal stacked bars of encode/decode/other per search, methods on the y-axis."""
    labels = list(means)
    positions = range(len(labels))

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.5))
    left = [0.0] * len(labels)
    for segment, color in zip(PLOT_SEGMENTS, PLOT_COLORS):
        values = [segment_value(means[label], segment) for label in labels]
        ax.barh(list(positions), values, left=left, color=color, label=segment)
        left = [base + value for base, value in zip(left, values)]

    # Annotate each bar with its total at the end.
    for position, total in zip(positions, left):
        ax.text(total, position, f"  {total:.2g} s", va="center", fontsize=7)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first method at the top
    ax.set_title(f"{title}  (search with {num_steps} steps)")
    ax.set_xlabel("Batch time per search (s)")
    vmax = max(left, default=1.0)
    ax.set_xlim(0, vmax * 1.15)  # headroom for the value labels
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig


@app.cell
def _(means: dict[str, dict[str, float]], num_steps, problem, size_ui):
    fig = plot_breakdown(means, f"{problem}{size_ui.value}", num_steps)
    fig
    return (fig,)


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save figure")
    save_button
    return (save_button,)


@app.cell
def _(OUT_DIR, fig, problem, save_button, size_ui):
    mo.stop(not save_button.value, mo.md("Click **Save figure** to write the PDF."))

    problem_dir = OUT_DIR / problem.lower()
    problem_dir.mkdir(parents=True, exist_ok=True)
    path = problem_dir / f"inference_breakdown_{problem.lower()}{size_ui.value}.pdf"
    save_figure(fig, path)
    mo.md(f"Saved to `{path}`")
    return


if __name__ == "__main__":
    app.run()
