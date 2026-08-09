from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple, Self

import torch

from core.problems.solution import BaseSolution

from .acceptance_criterion import (
    AcceptanceCriterion,
    AcceptanceCriterionConfig,
    SimulatedAnnealingAcceptance,
    SimulatedAnnealingAcceptanceConfig,
)
from .improvement_step import ImprovementStep


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
    exchange: AugmentationExchange | None = None,
    on_step: Callable[[int, S], None] | None = None,
    step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
) -> SearchResult:
    criterion.reset()
    best = solution
    history: list[float] = []
    with improvement_step.search_session(solution):
        for step in step_progress(range(n_iterations)):
            solution = improvement_step(solution, criterion)
            if exchange is not None:
                solution = exchange(solution)
            best = solution.where(solution.objective() < best.objective(), best)
            history.append(solution.objective().mean().item())
            criterion.step()
            if on_step is not None:
                on_step(step + 1, solution)
    return SearchResult(solution=solution, best_solution=best, history=history)


class AugmentationExchange:
    """Cross-augmentation exchange for augmented simulated annealing."""

    def __init__(self, criterion: SimulatedAnnealingAcceptance, threshold_factor: float) -> None:
        self._criterion = criterion
        self._threshold_factor = threshold_factor

    @classmethod
    def from_config(
        cls, config: AcceptanceCriterionConfig, criterion: AcceptanceCriterion
    ) -> Self | None:
        if (
            not isinstance(config, SimulatedAnnealingAcceptanceConfig)
            or not config.augmented_exchange.enabled
        ):
            return None
        assert isinstance(criterion, SimulatedAnnealingAcceptance), (
            f"Expected an SA criterion for {type(config).__name__}, got {type(criterion).__name__}"
        )
        return cls(criterion, config.augmented_exchange.threshold_factor)

    def __call__[S: BaseSolution](self, solution: S) -> S:
        costs = solution.objective()  # (B, A)
        best_cost = costs.min(dim=-1, keepdim=True).values  # (B, 1)
        threshold = best_cost + self._criterion.temperature * self._threshold_factor  # (B, 1)
        return self._exchange_above_threshold(solution, costs, threshold)

    @staticmethod
    def _exchange_above_threshold[S: BaseSolution](
        solution: S, costs: torch.Tensor, threshold: torch.Tensor
    ) -> S:
        is_bad = costs > threshold  # (B, A)
        has_any_good = (~is_bad).any(dim=-1, keepdim=True)  # (B, 1)

        noise = torch.rand_like(costs).masked_fill(is_bad, float("-inf"))
        random_good_idx = noise.argmax(dim=-1, keepdim=True)  # (B, 1)

        candidate = solution.gather(-1, random_good_idx).expand(*solution.batch_size)
        do_replace = is_bad & has_any_good
        return candidate.where(do_replace, solution)
