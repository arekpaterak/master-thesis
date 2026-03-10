import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Combinatorial Optimization Problems: Overview
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch

    from core.problems.archive.op import OPConfig, get_op_generator, visualize_op_instance
    from core.problems.cvrp import CVRPConfig, CVRPInstance, CVRPSolution, TensorDictCVRPSampler
    from core.problems.tsp import TensorTSPSampler, TSPConfig, TSPInstance
    from core.problems.tsp.solution import TSPSolution

    return (
        CVRPConfig,
        CVRPInstance,
        CVRPSolution,
        OPConfig,
        TSPConfig,
        TSPInstance,
        TSPSolution,
        TensorDictCVRPSampler,
        TensorTSPSampler,
        get_op_generator,
        plt,
        torch,
        visualize_op_instance,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Traveling Salesman Problem
    """)
    return


@app.cell
def _(TSPConfig, TSPInstance, TensorTSPSampler, plt):
    tsp_config = TSPConfig(num_locations=20)
    tsp_sampler = TensorTSPSampler(tsp_config)
    t = tsp_sampler(2)
    print(t[0])

    instance = TSPInstance.from_tensor(t[0])
    instance.visualize()
    plt.gca()
    return instance, tsp_config


@app.cell
def _(TSPSolution, instance, plt, torch, tsp_config):
    random_tour = torch.randperm(tsp_config.num_locations)
    example_tsp_solution = TSPSolution(instance, tour=random_tour)
    example_tsp_solution.visualize()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Capacitated Vehicle Problem
    """)
    return


@app.cell
def _(CVRPConfig, CVRPInstance, TensorDictCVRPSampler, plt):
    cvrp_config = CVRPConfig.for_n(20)
    cvrp_sampler = TensorDictCVRPSampler(cvrp_config)
    _td = cvrp_sampler(2)
    print(_td)
    print(_td[0])
    cvrp_instance = CVRPInstance.from_td(_td[0])
    cvrp_instance.visualize()
    plt.gca()
    return (cvrp_instance,)


@app.cell
def _(CVRPSolution, cvrp_instance, plt):
    from core.problems.cvrp.solve import (
        all_customers_one_route,
        index_sequential,
        one_customer_per_route,
    )

    fig, axes = plt.subplots(3, 1, figsize=(6, 18))

    strategies = [
        (one_customer_per_route, "One customer per route"),
        (all_customers_one_route, "All customers one route"),
        (index_sequential, "Index sequential"),
    ]

    for ax, (strategy, title) in zip(axes, strategies):
        solution: CVRPSolution = strategy(cvrp_instance)
        solution.visualize(
            ax=ax, with_problem_size=False, depot_in_legend=False, with_route_labels=False
        )
        ax.set_title(f"{title}\n{ax.get_title()}")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Orienteering Problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. Constant**
    """)
    return


@app.cell
def _(OPConfig, get_op_generator, plt, visualize_op_instance):
    _op_config = OPConfig(num_nodes=20, prize_type="const")
    _op_generator = get_op_generator(_op_config)
    _op_generator.device = "cpu"
    _td = _op_generator([1])
    print(_td)
    print(_td[0])
    _min_loc = _op_generator.min_loc
    _max_loc = _op_generator.max_loc
    visualize_op_instance(_td[0], _min_loc, _max_loc, prize_type="const")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **2. Uniform**
    """)
    return


@app.cell
def _(OPConfig, get_op_generator, plt, visualize_op_instance):
    _op_config = OPConfig(num_nodes=20, prize_type="unif")
    _op_generator = get_op_generator(_op_config)
    _op_generator.device = "cpu"
    _td = _op_generator([1])
    print(_td)
    _min_loc = _op_generator.min_loc
    _max_loc = _op_generator.max_loc
    _ = _td["locs"][0].cpu().numpy()
    _ = _td["depot"][0].cpu().numpy()
    _ = _td["prize"][0].cpu().numpy()
    _ = _td["max_length"][0].item()
    visualize_op_instance(_td[0], _min_loc, _max_loc, prize_type="unif")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **3. Distance**
    """)
    return


@app.cell
def _(OPConfig, get_op_generator, plt, visualize_op_instance):
    _op_config = OPConfig(num_nodes=20, prize_type="dist")
    _op_generator = get_op_generator(_op_config)
    _td = _op_generator([1])
    print(_td)
    _min_loc = _op_generator.min_loc
    _max_loc = _op_generator.max_loc
    visualize_op_instance(_td[0], _min_loc, _max_loc, prize_type="dist")
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
