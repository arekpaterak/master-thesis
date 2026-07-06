from __future__ import annotations

from collections.abc import Callable, Iterable

from core.lns.acceptance_criterion import SimulatedAnnealingAcceptance, build_criterion
from core.lns.search import build_augmentation_synchronizer, search
from core.methods.base import Method
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


def solve(
    method: Method,
    instance: BaseProblemInstance,
    *,
    on_step: Callable[[int, BaseSolution], None] | None = None,
    step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
) -> BaseSolution:
    criterion = build_criterion(
        method.search.acceptance, n_steps=method.search.num_improvement_steps
    )

    augmented = method.search.n_augmentations > 1
    if augmented:
        instance = instance.augment(method.search.n_augmentations)
    solution = method.generate_start_solution(instance)

    synchronize: Callable[[BaseSolution], BaseSolution] = lambda solution: solution
    if augmented and isinstance(criterion, SimulatedAnnealingAcceptance):
        synchronize = build_augmentation_synchronizer(
            criterion, method.search.augmented_threshold_factor
        )

    # Reduce a (B, A) augmented solution to (B,). Identity for the plain (B,) path.
    reduce_augmentations = _best_augmentation if augmented else lambda solution: solution

    step_callback: Callable[[int, BaseSolution], None] | None = None
    if on_step is not None:

        def step_callback(step: int, current: BaseSolution) -> None:
            on_step(step, reduce_augmentations(current))

        step_callback(0, solution)

    result = search(
        solution,
        improvement_step=method.improvement_step,
        criterion=criterion,
        n_iterations=method.search.num_improvement_steps,
        synchronize=synchronize,
        on_step=step_callback,
        step_progress=step_progress,
    )

    return reduce_augmentations(result.solution)


def _best_augmentation(solution: BaseSolution) -> BaseSolution:
    """Reduce a (B, A) augmented solution to (B,) by picking the best augmentation per instance."""
    best_idx = solution.objective().argmin(dim=-1, keepdim=True)
    return solution.gather(-1, best_idx).squeeze(-1)
