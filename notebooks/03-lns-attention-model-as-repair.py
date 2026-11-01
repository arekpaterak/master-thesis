import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    from contextlib import contextmanager
    from pathlib import Path

    import marimo as mo
    import torch

    from core.lns.acceptance_criterion import AcceptanceCriterion, GreedyAcceptance
    from core.lns.operators.destroy.random_segment import TSPRandomSegmentDestroy
    from core.lns.operators.repair.am_segment import AttentionModelRepair
    from core.lns.search import search
    from core.models.attention_model import AttentionModel
    from core.models.attention_model.statistical_test import one_sided_paired_ttest
    from core.models.attention_model.training import training_step
    from core.models.attention_model.training_helpers import prepare_baseline, sync_baseline_
    from core.models.utils import load_checkpoint, save_checkpoint
    from core.problems.solution import BaseSolution
    from core.problems.tsp import TSPDistribution, TSPSampler
    from core.problems.tsp.solve import random_permutation

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NOTEBOOK_PATH = Path(__file__).parent if "__file__" in dir() else Path(".")

    class DestroyRepairStep:
        def __init__(self, destroy, repair) -> None:
            self._destroy = destroy
            self._repair = repair

        @contextmanager
        def search_session(self, solution: BaseSolution):
            yield

        def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
            partial = self._destroy(solution)
            candidate = self._repair(partial)
            mask = accept(solution, candidate)
            return candidate.where(mask, solution)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Large Neighbourhood Search (LNS) with Attention Model as Repair Operator
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
    print(f"Using device: {DEVICE}")
    print(f"Notebook path: {NOTEBOOK_PATH}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Train the AM to be used as the repair operator in the LNS.

    If a checkpoint already exists it is loaded automatically. Otherwise, click
    **Start training** to train the model (this is skipped until you press it).
    """)
    return


@app.cell
def _():
    train_button = mo.ui.run_button(label="Start training")
    train_button
    return (train_button,)


@app.cell
def _(train_button):
    tsp_sampler = TSPSampler(
        TSPDistribution(
            num_locations=20,
            min_coordinate=0.0,
            max_coordinate=1.0,
            distribution_of_locations="uniform",
        ),
    )

    model = AttentionModel().to(DEVICE)

    model_path = NOTEBOOK_PATH / "checkpoints" / "attention_model_for_repair.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        print(f"Loading model from {model_path}")
        load_checkpoint(path=model_path, model=model)
    elif not train_button.value:
        print("No checkpoint found. Click 'Start training' to train the model.")
    else:
        n_epochs = 15
        steps_per_epoch = 100
        batch_size = 512
        lr = 1e-4
        alpha = 0.05

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        baseline = prepare_baseline(model)

        for epoch in range(1, n_epochs + 1):
            for _ in range(steps_per_epoch):
                instance = tsp_sampler(batch_size).to(DEVICE)
                training_step(
                    model=model, baseline=baseline, optimizer=optimizer, instance=instance
                )

            test = one_sided_paired_ttest(
                model,
                baseline,
                tsp_sampler,
                n_instances=1_000,
                batch_size=512,
                alpha=alpha,
                device=DEVICE,
            )
            if test.improved:
                sync_baseline_(baseline, model)
            print(
                f"Epoch {epoch:2d}: "
                f"obj_model={test.obj_model_mean:.4f}, "
                f"obj_baseline={test.obj_baseline_mean:.4f}, "
                f"updated={test.improved}"
            )

        save_checkpoint(
            path=model_path,
            model=model,
        )
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Basic LNS
    """)
    return


@app.cell
def _():
    run_experiments_button = mo.ui.run_button(label="Run experiments")
    run_experiments_button
    return (run_experiments_button,)


@app.cell
def _(model, run_experiments_button):
    mo.stop(
        not run_experiments_button.value,
        mo.md("Click **Run experiments** to run the LNS experiments."),
    )

    experiments = [
        (20, 20, 1),
        (20, 10, 50),
        (20, 5, 100),
        (50, 15, 50),
        # (100, 15, 50),
    ]

    results_of_experiments = []
    for num_locations, destroy_size, n_iterations in experiments:
        print(
            f"Experiment: num_locations={num_locations}, destroy_size={destroy_size}, n_iterations={n_iterations}"
        )

        destroy = TSPRandomSegmentDestroy(destroy_size)
        repair = AttentionModelRepair(model)
        accept = GreedyAcceptance()

        sampler = TSPSampler(
            TSPDistribution(
                num_locations=num_locations,
                min_coordinate=0.0,
                max_coordinate=1.0,
                distribution_of_locations="uniform",
            ),
            seed=42,
        )
        batch = sampler(512).to(DEVICE)

        initial_solutions = random_permutation(batch)

        search_result = search(
            initial_solutions,
            improvement_step=DestroyRepairStep(destroy, repair),
            criterion=accept,
            n_iterations=n_iterations,
        )

        results_of_experiments.append(
            {
                "num_locations": num_locations,
                "destroy_size": destroy_size,
                "n_iterations": n_iterations,
                "mean_initial_cost": initial_solutions.objective().mean().item(),
                "mean_final_cost": search_result.best_solution.objective().mean().item(),
                "history": search_result.history,
            }
        )
    return (results_of_experiments,)


@app.cell
def _(results_of_experiments):
    results_of_experiments
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training the AM within the LNS loop
    """)
    return


@app.cell
def _():
    # TODO
    return


if __name__ == "__main__":
    app.run()
