from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple

from core.problems.solution import BaseSolution

from .acceptance_criterion import AcceptanceCriterion
from .improvement_step import ImprovementStep
from .utils import select_solution


class SearchResult(NamedTuple):
    solution: BaseSolution
    best_solution: BaseSolution
    history: list[float]


def search[S: BaseSolution](
    solution: S,
    *,
    improvement_step: ImprovementStep,
    criterion: AcceptanceCriterion,
    n_iterations: int,
    on_step: Callable[[int, S], None] | None = None,
    step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
) -> SearchResult:
    criterion.reset()
    best = solution
    history: list[float] = []
    for step in step_progress(range(n_iterations)):
        solution = improvement_step(solution, criterion)
        best = select_solution(solution.objective() < best.objective(), solution, best)
        history.append(solution.objective().mean().item())
        criterion.step()
        if on_step is not None:
            on_step(step + 1, solution)
    return SearchResult(solution=solution, best_solution=best, history=history)
