import marimo

__generated_with = "0.22.5"
app = marimo.App()

with app.setup:
    import torch


@app.cell
def _():
    from core.problems.tsp import TSPConfig, TSPSampler
    from core.problems.tsp.solve import random_permutation

    n = 10
    batch_size = 4

    tsp_instance = TSPSampler(TSPConfig(num_locations=n), seed=42)(batch_size)
    tsp_solution = random_permutation(tsp_instance)
    tsp_solution.tour
    return n, tsp_solution


@app.cell
def _(tsp_solution):
    from core.lns.operators.destroy.random_node import RandomNodeDestroy

    tsp_destroyed_solution = RandomNodeDestroy(5, seed=42)(tsp_solution)
    tsp_destroyed_solution.tour
    return (tsp_destroyed_solution,)


@app.cell
def _(tsp_destroyed_solution):
    tsp_destroyed_solution.unassigned_nodes()
    return


@app.cell
def _(tsp_destroyed_solution):
    from core.lns.operators.repair import RandomRepair

    random_repair = RandomRepair(seed=42)
    _repaired = random_repair(tsp_destroyed_solution)

    print(_repaired.tour)
    print(_repaired.objective())
    return


@app.cell
def _(tsp_destroyed_solution):
    from core.lns.operators.repair import GreedyRepair

    greedy_repair = GreedyRepair()
    _repaired = greedy_repair(tsp_destroyed_solution)

    print(_repaired.tour)
    print(_repaired.objective())
    return


@app.cell
def _(tsp_solution):
    tsp_solution.neighbours_index()
    return


@app.cell
def _(n, tsp_solution):
    # seq: [B, L]
    B, L = tsp_solution.tour.shape
    batch_idx = torch.arange(B).unsqueeze(1)  # [B, 1]

    nxt = torch.full((B, n), -1)
    prv = torch.full((B, n), -1)

    nxt[batch_idx, tsp_solution.tour] = tsp_solution.tour.roll(-1, dims=1)
    prv[batch_idx, tsp_solution.tour] = tsp_solution.tour.roll(1, dims=1)

    print(tsp_solution.tour)
    print(nxt)
    print(prv)
    return


if __name__ == "__main__":
    app.run()
