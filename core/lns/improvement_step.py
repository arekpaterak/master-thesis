from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.apply_and_repair import ApplyAndRepair, build_apply_and_repair
from core.lns.operators.destroy.base import DestroyOperator
from core.problems.solution import BaseSolution

if TYPE_CHECKING:
    from core.lns.config import SearchParams

Mode = Literal["parallel", "sequential"]


class ImprovementStep:
    def __init__(
        self,
        *,
        destroy: DestroyOperator,
        apply_and_repair: ApplyAndRepair,
        mode: Mode,
        num_rollouts: int,
    ) -> None:
        if destroy.selection_type is not apply_and_repair.selection_type:
            raise ValueError(
                f"destroy emits {destroy.selection_type.__name__} but apply_and_repair consumes "
                f"{apply_and_repair.selection_type.__name__}; they must match."
            )
        self._destroy = destroy
        self._apply_and_repair = apply_and_repair
        self._mode = mode
        self._num_rollouts = num_rollouts

    @contextmanager
    def search_session(self, solution: BaseSolution) -> Iterator[None]:
        with self._destroy.search_session(solution):
            yield

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        selection = self._destroy.select(solution, self._num_rollouts)
        if self._mode == "sequential":
            return self._apply_and_repair.sequential(solution, selection, accept)

        candidates = self._apply_and_repair.parallel(solution, selection)  # (*B, K)
        best_idx = candidates.objective().argmin(dim=-1, keepdim=True)
        best = candidates.gather(-1, best_idx).squeeze(-1)
        mask = accept(solution, best)
        return best.where(mask, solution)


class NoopImprovementStep(ImprovementStep):
    def __init__(self) -> None:
        pass

    @contextmanager
    def search_session(self, solution: BaseSolution) -> Iterator[None]:
        yield

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        return solution


def _mode_and_backend(improvement_step_type: str) -> tuple[Mode, str]:
    match improvement_step_type:
        case "sequential":
            return "sequential", "torch"
        case "parallel":
            return "parallel", "torch"
        case "numba_sequential":
            return "sequential", "numba"
        case "numba_parallel":
            return "parallel", "numba"
        case _:
            raise NotImplementedError(f"Unknown improvement step type: {improvement_step_type!r}")


def build_improvement_step(
    search: SearchParams,
    *,
    destroy: DestroyOperator,
    problem_name: str,
    seed: int = 0,
) -> ImprovementStep:
    mode, backend = _mode_and_backend(search.improvement_step_type)
    apply_and_repair = build_apply_and_repair(
        search, problem_name=problem_name, backend=backend, seed=seed
    )
    return ImprovementStep(
        destroy=destroy,
        apply_and_repair=apply_and_repair,
        mode=mode,
        num_rollouts=search.num_rollouts,
    )


# TODO: Is it the best place for that? Maybe it should be cls method
def build_random_node_destroy_step(
    search: SearchParams, *, problem_name: str, seed: int | None
) -> ImprovementStep:
    from core.lns.operators.destroy.random_node import RandomNodeDestroy

    destroy = RandomNodeDestroy(destruction_size=search.deconstruction_size, seed=seed)
    return build_improvement_step(
        search, destroy=destroy, problem_name=problem_name, seed=seed or 0
    )
