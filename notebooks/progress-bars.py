# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Nested progress bars demo

    Three levels: **epochs**, **batches**, **improvement steps**.

    The epoch bar persists for the whole run. The batch and improvement-step
    bars use `remove_on_exit=True`, so they disappear when their loop
    finishes and a fresh bar appears for the next iteration of the parent
    loop. All loops live in one cell so the bars share a single output
    region.
    """)
    return


@app.cell
def _():
    num_epochs = mo.ui.slider(1, 10, value=3, label="epochs")
    num_batches = mo.ui.slider(1, 20, value=4, label="batches per epoch")
    num_improvement_steps = mo.ui.slider(1, 50, value=8, label="improvement steps per batch")
    sleep_seconds = mo.ui.slider(0.0, 0.2, step=0.01, value=0.05, label="sleep per step (s)")
    mo.vstack([num_epochs, num_batches, num_improvement_steps, sleep_seconds])
    return num_batches, num_epochs, num_improvement_steps, sleep_seconds


@app.cell
def _():
    run_button = mo.ui.run_button(label="Run demo")
    run_button
    return (run_button,)


@app.cell
def _(
    num_batches,
    num_epochs,
    num_improvement_steps,
    run_button,
    sleep_seconds,
):
    is_script_mode = mo.app_meta().mode == "script"
    should_run = is_script_mode or run_button.value

    if should_run:
        for epoch in mo.status.progress_bar(
            range(num_epochs.value),
            title="Epochs",
            completion_title="Training done",
        ):
            for batch in mo.status.progress_bar(
                range(num_batches.value),
                title=f"Epoch {epoch} - batches",
                remove_on_exit=True,
            ):
                with mo.status.progress_bar(
                    total=num_improvement_steps.value,
                    title=f"Epoch {epoch} - batch {batch} - improvement steps",
                    remove_on_exit=True,
                ) as improvement_bar:
                    for _step in range(num_improvement_steps.value):
                        time.sleep(sleep_seconds.value)
                        improvement_bar.update()
    return


if __name__ == "__main__":
    app.run()
