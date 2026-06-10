from __future__ import annotations

from collections.abc import Callable

from core.problems.cvrp.solve import one_customer_per_route_graph
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution
from core.problems.tsp.solve import random_permutation_graph

GenerateInitialSolutionFn = Callable[[BaseProblemInstance], BaseSolution]


def resolve_start_solution(problem_name: str) -> GenerateInitialSolutionFn:
    match problem_name:
        case "cvrp":
            return one_customer_per_route_graph
        case "tsp":
            return random_permutation_graph
        case _:
            raise ValueError(f"Unsupported problem name: {problem_name}")
