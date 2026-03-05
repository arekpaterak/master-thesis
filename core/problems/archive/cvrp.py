from matplotlib import pyplot as plt
from rl4co.envs.routing.cvrp.generator import CVRPGenerator
from tensordict import TensorDict
from torch.distributions import Uniform

from core.problems.cvrp import CVRPConfig


def get_cvrp_generator(config: CVRPConfig) -> CVRPGenerator:
    num_nodes = config.num_locations
    min_loc = config.min_coordinate
    max_loc = config.max_coordinate
    min_demand = config.min_demand
    max_demand = config.max_demand
    vehicle_capacity = config.vehicle_capacity

    return CVRPGenerator(
        num_loc=num_nodes,
        min_loc=min_loc,
        max_loc=max_loc,
        loc_distribution=Uniform,
        min_demand=min_demand,
        max_demand=max_demand,
        vehicle_capacity=vehicle_capacity,
    )


def visualize_cvrp_instance(
    instance: TensorDict, min_loc: float, max_loc: float, max_demand: float
) -> tuple[plt.Figure, plt.Axes]:
    assert instance.batch_dims == 0, "Can only visualize a single instance."

    locations = instance["locs"].cpu().numpy()
    demands = instance["demand"].cpu().numpy()
    depot = instance["depot"].cpu().numpy()
    _ = instance["capacity"].item()

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        locations[:, 0],
        locations[:, 1],
        c=demands,
        cmap="Reds",
        s=100,
        vmin=0,
        vmax=max_demand,
        edgecolors="black",
        linewidths=1,
    )

    ax.scatter(
        depot[0],
        depot[1],
        c="orange",
        s=100,
        marker="s",
        label="depot",
        edgecolors="black",
        linewidths=1,
    )

    plt.colorbar(scatter, label="Demand")
    ax.set_xlim(min_loc - 0.1, max_loc + 0.1)
    ax.set_ylim(min_loc - 0.1, max_loc + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    ax.set_title(f"CVRP({len(locations)})")
    ax.grid(True, linestyle="--", alpha=0.4)

    return fig, ax
