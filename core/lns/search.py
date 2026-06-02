from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch

from core.problems.solution import BaseSolution

from .acceptance_criterion import AcceptanceCriterion, SimulatedAnnealingAcceptance
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


def augmented_search[S: BaseSolution](
    augmented_solution: S,
    *,
    improvement_step: ImprovementStep,
    criterion: SimulatedAnnealingAcceptance,
    n_iterations: int,
    threshold_factor: float,
    on_step: Callable[[int, S], None] | None = None,
    step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
) -> SearchResult:
    """Algorithm 2 (Augmented Simulated Annealing) from the NDS paper.

    Runs LNS independently on ``A`` D4-augmented copies of each instance and at
    each outer iteration replaces high-cost augmented solutions by randomly
    sampled good ones whose cost is within a temperature-controlled threshold.

    The augmentation dim must be the innermost batch dim, so cost reductions and
    cross-augmentation exchange use ``dim=-1``.

    Args:
        augmented_solution: Starting solution with shape ``(B, A, ...)``.
        improvement_step: Inner LNS step that takes a solution + criterion.
        criterion: SA criterion. Its current temperature drives the exchange
            threshold; must be SA (not greedy) because the threshold depends
            on temperature.
        n_iterations: Number of outer iterations.
        threshold_factor: ``delta`` in the paper. Higher = more aggressive
            replacement when temperature is high.
        on_step: Optional callback invoked after each outer iteration.
        step_progress: Optional progress iterator.

    Returns:
        SearchResult with the final augmented solution, the per-(B, A) best
        observed solution, and a history of the mean over B of min-over-A cost.
    """
    criterion.reset()
    best = augmented_solution
    history: list[float] = []
    for step in step_progress(range(n_iterations)):
        augmented_solution = improvement_step(augmented_solution, criterion)
        costs = augmented_solution.objective()  # (B, A)
        best_cost = costs.min(dim=-1, keepdim=True).values  # (B, 1)
        threshold = best_cost + criterion.temperature * threshold_factor  # (B, 1)
        augmented_solution = _exchange_above_threshold(augmented_solution, costs, threshold)

        best = select_solution(costs < best.objective(), augmented_solution, best)
        history.append(costs.min(dim=-1).values.mean().item())
        criterion.step()
        if on_step is not None:
            on_step(step + 1, augmented_solution)
    return SearchResult(solution=augmented_solution, best_solution=best, history=history)


def _exchange_above_threshold[S: BaseSolution](
    solution: S, costs: torch.Tensor, threshold: torch.Tensor
) -> S:
    """Replace each above-threshold solution by a uniformly random below-threshold one.

    Operates per-instance over the innermost (augmentation) batch dim. If no
    below-threshold solution exists for an instance, that instance is left
    untouched.
    """
    is_bad = costs > threshold  # (B, A)
    has_any_good = (~is_bad).any(dim=-1, keepdim=True)  # (B, 1)

    noise = torch.rand_like(costs).masked_fill(is_bad, float("-inf"))
    random_good_idx = noise.argmax(dim=-1, keepdim=True)  # (B, 1)

    candidate = solution.gather(-1, random_good_idx).expand(*solution.batch_size)
    do_replace = is_bad & has_any_good
    return select_solution(do_replace, candidate, solution)
