from __future__ import annotations

from collections.abc import Callable

from core.lns.config import RepairType
from core.lns.operators.repair import NoopRepair, RepairOperator
from core.lns.operators.repair.greedy_insert import CVRPGreedyInsertRepair
from core.lns.operators.repair.greedy_reconnect import (
    CVRPGreedyReconnectRepair,
    TSPGreedyReconnectRepair,
)
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


def resolve_start_solution(problem_name: str) -> GenerateInitialSolutionFn:
    match problem_name:
        case "cvrp":
            return one_customer_per_route_graph
        case "tsp":
            return random_permutation_graph
        case _:
            raise ValueError(f"Unsupported problem name: {problem_name}")


# TODO: Move under lns/operators/repair/common.py
def resolve_repair(
    problem_name: str,
    repair_type: RepairType,
    *,
    deconstruction_size: int | None,
) -> RepairOperator:
    if repair_type == "noop":
        return NoopRepair()
    if repair_type == "greedy_insert":
        if problem_name != "cvrp":
            raise ValueError(f"greedy_insert is only implemented for CVRP, got {problem_name!r}")
        return CVRPGreedyInsertRepair(deconstruction_size=deconstruction_size)
    match problem_name:
        case "cvrp":
            return CVRPGreedyReconnectRepair(deconstruction_size=deconstruction_size)
        case "tsp":
            return TSPGreedyReconnectRepair(deconstruction_size=deconstruction_size)
        case _:
            raise ValueError(f"Unsupported problem name: {problem_name}")
