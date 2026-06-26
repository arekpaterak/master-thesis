from __future__ import annotations

from collections.abc import Callable

from core.lns.acceptance_criterion import GreedyAcceptance
from core.lns.config import SearchParams
from core.lns.improvement_step import build_random_destroy_step
from core.lns.search import search as lns_search
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution

StartSolutionFn = Callable[[BaseProblemInstance], BaseSolution]


def with_warmup(
    generate_start_solution: StartSolutionFn,
    *,
    problem_name: str,
    search: SearchParams,
) -> StartSolutionFn:
    params = search.start_solution_params
    if params.num_improvement_steps <= 0:
        return generate_start_solution

    # The warmup is a single-rollout, single-repair loop (the original create_starting_solution);
    # everything else (repair_type, improvement_step_type) is inherited from the parent search.
    warmup_search = search.model_copy(
        update={
            "deconstruction_size": params.deconstruction_size,
            "num_rollouts": 1,
            "n_repairs": 1,
        }
    )
    warmup_step = build_random_destroy_step(warmup_search, problem_name=problem_name, seed=0)

    def wrapped(instance: BaseProblemInstance) -> BaseSolution:
        solution = generate_start_solution(instance)
        return lns_search(
            solution,
            improvement_step=warmup_step,
            criterion=GreedyAcceptance(),
            n_iterations=params.num_improvement_steps,
        ).solution

    return wrapped
