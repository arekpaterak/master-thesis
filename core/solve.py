from __future__ import annotations

from collections.abc import Callable, Iterable

from core.lns.acceptance_criterion import SimulatedAnnealingAcceptance, build_criterion
from core.lns.search import augmented_search, search
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

    # TODO: Possibly refactor by unifying paths
    if method.search.n_augmentations > 1:
        if not isinstance(criterion, SimulatedAnnealingAcceptance):
            raise ValueError(
                "Augmented search (n_augmentations > 1) requires a simulated_annealing "
                "acceptance criterion -- the cross-augmentation threshold depends on "
                "the SA temperature."
            )
        instance = instance.augment(method.search.n_augmentations)
        solution = method.generate_start_solution(instance)
        step_callback = None
        if on_step is not None:

            def step_callback(step: int, aug_solution: BaseSolution) -> None:
                on_step(step, _best_augmentation(aug_solution))

            step_callback(0, solution)
        result = augmented_search(
            solution,
            improvement_step=method.improvement_step,
            criterion=criterion,
            n_iterations=method.search.num_improvement_steps,
            threshold_factor=method.search.augmented_threshold_factor,
            on_step=step_callback,
            step_progress=step_progress,
        )
        return _best_augmentation(result.solution)
    else:
        solution = method.generate_start_solution(instance)
        if on_step is not None:
            on_step(0, solution)
        result = search(
            solution,
            improvement_step=method.improvement_step,
            criterion=criterion,
            n_iterations=method.search.num_improvement_steps,
            on_step=on_step,
            step_progress=step_progress,
        )
        return result.solution


def _best_augmentation(solution: BaseSolution) -> BaseSolution:
    """Reduce a (B, A) augmented solution to (B,) by picking the best augmentation per instance."""
    best_idx = solution.objective().argmin(dim=-1, keepdim=True)
    return solution.gather(-1, best_idx).squeeze(-1)
