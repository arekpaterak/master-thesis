from __future__ import annotations

from collections.abc import Callable

from core.lns.acceptance_criterion import GreedyAcceptance
from core.lns.components import LNSComponents
from core.lns.config import SearchParams
from core.lns.improvement_step import build_random_node_destroy_step
from core.lns.search import search as lns_search
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution

StartSolutionFn = Callable[[BaseProblemInstance], BaseSolution]


def with_warmup(
    generate_start_solution: StartSolutionFn,
    *,
    lns: LNSComponents,
    search: SearchParams,
) -> StartSolutionFn:
    params = search.start_solution_params
    if params.num_improvement_steps <= 0:
        return generate_start_solution

    # The warmup is a single-rollout, single-repair loop.
    warmup_search = search.model_copy(
        update={
            "deconstruction_size": params.deconstruction_size,
            "num_rollouts": 1,
            "n_repairs": 1,
            "insert_in_affected_routes_only": params.insert_in_affected_routes_only,
            "first_repair_uses_rollout_order": params.first_repair_uses_rollout_order,
        }
    )
    warmup_step = build_random_node_destroy_step(warmup_search, lns=lns, seed=0)

    def wrapped(instance: BaseProblemInstance) -> BaseSolution:
        solution = generate_start_solution(instance)
        return lns_search(
            solution,
            improvement_step=warmup_step,
            criterion=GreedyAcceptance(),
            n_iterations=params.num_improvement_steps,
        ).solution

    return wrapped
