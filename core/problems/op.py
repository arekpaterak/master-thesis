from dataclasses import dataclass
from typing import Literal

from matplotlib import pyplot as plt
from rl4co.envs.routing.op.generator import OPGenerator
from tensordict import TensorDict
from torch.distributions import Uniform

PrizeType = Literal["const", "unif", "dist"]


@dataclass(frozen=True)
class OPConfig:
    num_nodes: int
    prize_type: PrizeType
    min_loc: float = 0.0
    max_loc: float = 1.0
    max_length: float | None = None


def get_op_generator(config: OPConfig) -> OPGenerator:
    num_nodes = config.num_nodes
    min_loc = config.min_loc
    max_loc = config.max_loc
    prize_type = config.prize_type
    max_length = config.max_length

    kwargs = {
        "num_loc": num_nodes,
        "min_loc": min_loc,
        "max_loc": max_loc,
        "loc_distribution": Uniform,
        "prize_type": prize_type,
        "prize_distribution": "dist",  # Hack
    }

    if max_length is not None:
        kwargs["max_length"] = max_length

    return OPGenerator(**kwargs)


def visualize_op_instance(
    instance: TensorDict,
    min_loc: float,
    max_loc: float,
    prize_type: PrizeType,
) -> tuple[plt.Figure, plt.Axes]:
    assert instance.batch_dims == 0, "Can only visualize a single instance."

    locations = instance["locs"].cpu().numpy()

    depot = instance["depot"].cpu().numpy()
    prizes = instance["prize"].cpu().numpy()

    max_length = instance["max_length"].item()

    if prize_type == "const":
        vmin, vmax = 1.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        locations[:, 0],
        locations[:, 1],
        c=prizes,
        cmap="Greens",
        s=100,
        vmin=vmin,
        vmax=vmax,
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
    cbar = plt.colorbar(scatter, label="Prize")
    if prize_type == "const":
        cbar.set_ticks([1.0])
    ax.set_xlim(min_loc - 0.1, max_loc + 0.1)
    ax.set_ylim(min_loc - 0.1, max_loc + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    ax.set_title(f"OP({len(locations)}, {max_length=})")
    ax.grid(True, linestyle="--", alpha=0.4)

    return fig, ax
