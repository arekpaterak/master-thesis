from __future__ import annotations

from collections.abc import Callable

from core.lns.operators.repair import NoopRepair, RepairOperator
from core.lns.operators.repair.greedy_reconnect import (
    CVRPGreedyReconnectRepair,
    TSPGreedyReconnectRepair,
)
from core.models.nds.functions import only_expanding_apply_rollout
from core.models.rollout import Rollout
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route_graph
from core.problems.instance import BaseProblemInstance
from core.problems.sampler import Sampler
from core.problems.solution import BaseSolution
from core.problems.tsp import TSPConfig, TSPSampler
from core.problems.tsp.solve import random_permutation_graph


def make_sampler(problem_name: str, size: int, *, seed: int | None = None) -> Sampler:
    match problem_name:
        case "cvrp":
            return CVRPSampler(CVRPConfig.for_n(size), seed=seed)
        case "tsp":
            return TSPSampler(TSPConfig(num_locations=size), seed=seed)
        case _:
            raise ValueError(f"Unsupported problem name: {problem_name}")


GenerateInitialSolutionFn = Callable[[BaseProblemInstance], BaseSolution]
ApplyRolloutFn = Callable[[BaseSolution, Rollout], BaseSolution]


def get_functions_by_problem(
    problem_name: str,
    *,
    noop: bool = False,
    deconstruction_size: int | None = None,
) -> tuple[GenerateInitialSolutionFn, RepairOperator, ApplyRolloutFn | None]:
    # Passing ``deconstruction_size`` lets the repair operator skip the
    # host-device sync that would otherwise size its inner loop.
    match problem_name:
        case "cvrp":
            generate_start_solution: GenerateInitialSolutionFn = one_customer_per_route_graph
            repair_fn: RepairOperator = CVRPGreedyReconnectRepair(
                deconstruction_size=deconstruction_size
            )
        case "tsp":
            generate_start_solution = random_permutation_graph
            repair_fn = TSPGreedyReconnectRepair(deconstruction_size=deconstruction_size)
        case _:
            raise ValueError(f"Unsupported problem name: {problem_name}")

    if noop:
        return generate_start_solution, NoopRepair(), only_expanding_apply_rollout

    return generate_start_solution, repair_fn, None
