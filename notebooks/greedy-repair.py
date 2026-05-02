import marimo

__generated_with = "0.22.5"
app = marimo.App()

with app.setup:
    from core.lns.operators.destroy import RandomEdgeDestroy, RandomNodeDestroy
    from core.lns.operators.repair import GreedyNodeRepair, RandomRepair
    from core.problems.tsp import TSPConfig, TSPSampler
    from core.problems.tsp.solve import random_permutation


@app.cell
def _():
    n = 4
    batch_size = 2

    tsp_instance = TSPSampler(TSPConfig(num_locations=n), seed=42)(batch_size)
    tsp_solution = random_permutation(tsp_instance)

    print(tsp_solution.tour)
    return tsp_instance, tsp_solution


@app.cell
def _(tsp_instance, tsp_solution):
    from core.problems.tsp.graph_solution import TSPGraphSolution

    tsp_graph_solution = TSPGraphSolution.create(instance=tsp_instance, sequence=tsp_solution.tour)

    print(tsp_graph_solution.next_nodes)
    return (tsp_graph_solution,)


@app.cell
def _(tsp_solution):
    destroyed_tsp_solution = RandomNodeDestroy(2, seed=42)(tsp_solution)
    destroyed_tsp_solution.tour
    return (destroyed_tsp_solution,)


@app.cell
def _(destroyed_tsp_solution):
    destroyed_tsp_solution.unassigned_nodes()
    return


@app.cell
def _(tsp_graph_solution):
    destroyed_tsp_graph_solution = RandomEdgeDestroy(4, seed=42)(tsp_graph_solution)
    destroyed_tsp_graph_solution.next_nodes
    return


@app.cell
def _(destroyed_tsp_solution):
    random_repair = RandomRepair(seed=42)
    _repaired = random_repair(destroyed_tsp_solution)

    print(_repaired.tour)
    print(_repaired.objective())
    return


@app.cell
def _(destroyed_tsp_solution):
    greedy_repair = GreedyNodeRepair()
    _repaired = greedy_repair(destroyed_tsp_solution)

    print(_repaired.tour)
    print(_repaired.objective())
    return


if __name__ == "__main__":
    app.run()
