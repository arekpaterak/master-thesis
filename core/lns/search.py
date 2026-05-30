from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from core.problems.solution import BaseSolution

from .acceptance_criterion import AcceptanceCriterion
from .improvement_step import ImprovementStep


class SearchResult(NamedTuple):
    solution: BaseSolution
    history: list[float]


def search[S: BaseSolution](
    solution: S,
    *,
    improvement_step: ImprovementStep,
    criterion: AcceptanceCriterion,
    n_iterations: int,
    on_step: Callable[[int, S], None] | None = None,
) -> SearchResult:
    criterion.reset()
    history: list[float] = []
    for step in range(n_iterations):
        solution = improvement_step(solution, criterion)
        history.append(solution.objective().mean().item())
        criterion.step()
        if on_step is not None:
            on_step(step + 1, solution)
    return SearchResult(solution=solution, history=history)
