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
    synchronize: Callable[[S], S] = lambda solution: solution,
    on_step: Callable[[int, S], None] | None = None,
    step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
) -> SearchResult:
    """Run the LNS destroy/repair/accept loop.

    Works on both plain ``(B, ...)`` and D4-augmented ``(B, A, ...)`` solutions;
    objective comparisons and ``select_solution`` are elementwise over the batch
    dims either way.

    Args:
        synchronize: Per-iteration hook applied after the improvement step and
            before best selection, mapping ``(B, A, ...) -> (B, A, ...)``. Defaults
            to the identity. For augmented simulated annealing (Algorithm 2 from the
            NDS paper) pass the closure from ``build_augmentation_synchronizer`` to
            exchange high-cost augmented copies for good ones within a
            temperature-controlled threshold.
    """
    criterion.reset()
    best = solution
    history: list[float] = []
    for step in step_progress(range(n_iterations)):
        solution = improvement_step(solution, criterion)
        solution = synchronize(solution)
        best = select_solution(solution.objective() < best.objective(), solution, best)
        history.append(solution.objective().mean().item())
        criterion.step()
        if on_step is not None:
            on_step(step + 1, solution)
    return SearchResult(solution=solution, best_solution=best, history=history)


def build_augmentation_synchronizer[S: BaseSolution](
    criterion: SimulatedAnnealingAcceptance, threshold_factor: float
) -> Callable[[S], S]:
    """Build the cross-augmentation exchange hook for augmented simulated annealing.

    At each iteration it replaces high-cost augmented copies by randomly sampled
    good ones whose cost is within a temperature-controlled threshold. The
    augmentation dim must be the innermost batch dim, so cost reductions and the
    exchange use ``dim=-1``.

    Args:
        criterion: SA criterion. Its current temperature (read live) drives the
            exchange threshold; must be SA because the threshold depends on
            temperature.
        threshold_factor: ``delta`` in the paper. Higher = more aggressive
            replacement when temperature is high.

    Returns:
        A ``(B, A, ...) -> (B, A, ...)`` hook to pass as ``search(synchronize=...)``.
    """

    def synchronize(solution: S) -> S:
        costs = solution.objective()  # (B, A)
        best_cost = costs.min(dim=-1, keepdim=True).values  # (B, 1)
        threshold = best_cost + criterion.temperature * threshold_factor  # (B, 1)
        return _exchange_above_threshold(solution, costs, threshold)

    return synchronize


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
