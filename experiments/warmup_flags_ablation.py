"""Ablation of the warmup repair flags on the CVRP-100 start solution.

Measures how ``insert_in_affected_routes_only`` and ``first_repair_uses_rollout_order``
change the start solution produced by the random-node-destroy warmup, on the same
validation set used for training (``val/objective_initial`` in wandb).

The warmup is: base construction (one customer per route) followed by
``num_improvement_steps`` iterations of random-node destroy + greedy-insert repair
under greedy acceptance -- exactly what ``core.lns.start_solution.with_warmup`` does,
but re-wired here so we can vary the two flags.

Run: uv run python experiments/warmup_flags_ablation.py
"""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import torch

import core.problems.cvrp.solve  # noqa: F401  (registers the CVRP initial solution)
from core.lns.config import SearchParams, StartSolutionParams
from core.lns.start_solution import with_warmup
from core.problems.dataset import load_dataset
from core.problems.solve import generate_initial_solution

PROBLEM_NAME = "cvrp"
DATASET_PATH = Path("data/synthetic/cvrp/vrp100_valid_seed1233.pkl")
NUM_INSTANCES = 1000

# The `search` section of nds_cvrp_100+ar.yml, minus the two flags we ablate.
BASE_SEARCH = dict(
    repair_type="greedy_insert",
    improvement_step_type="numba_sequential",
    apply_and_repair_backend="numba",
    deconstruction_size=15,
    num_rollouts=128,
    n_repairs=5,
)


def mean_objective_and_time(
    instance, *, insert_in_affected_routes_only: bool, first_repair_uses_rollout_order: bool
) -> tuple[float, float]:
    """Reproduce with_warmup for one flag combination; return (mean objective, seconds)."""
    search = SearchParams(
        **BASE_SEARCH,
        start_solution_params=StartSolutionParams(
            num_improvement_steps=50,
            deconstruction_size=15,
            destroy_type="random_node",
            insert_in_affected_routes_only=insert_in_affected_routes_only,
            first_repair_uses_rollout_order=first_repair_uses_rollout_order,
        ),
    )
    generate_start_solution = with_warmup(
        generate_initial_solution, problem_name=PROBLEM_NAME, search=search
    )
    start = time.perf_counter()
    final = generate_start_solution(instance)
    elapsed = time.perf_counter() - start
    return final.objective().mean().item(), elapsed


def main() -> None:
    torch.manual_seed(0)
    instance = load_dataset(PROBLEM_NAME, DATASET_PATH)[:NUM_INSTANCES]
    print(f"Loaded {instance.batch_size[0]} CVRP-{instance.num_customers} instances\n")

    # Trigger numba JIT compilation on a small slice so it is excluded from the timings.
    print("Trigger numba JIT...")
    for affected in (False, True):
        mean_objective_and_time(
            instance[:8],
            insert_in_affected_routes_only=affected,
            first_repair_uses_rollout_order=True,
        )
    print()

    header = f"{'insert_in_affected_routes_only':>31} | {'first_repair_uses_rollout_order':>31} | {'mean objective':>14} | {'time (s)':>8}"
    print(header)
    print("-" * len(header))
    for affected, order in product([False, True], [False, True]):
        objective, seconds = mean_objective_and_time(
            instance,
            insert_in_affected_routes_only=affected,
            first_repair_uses_rollout_order=order,
        )
        print(f"{str(affected):>31} | {str(order):>31} | {objective:>14.4f} | {seconds:>8.1f}")


if __name__ == "__main__":
    main()
