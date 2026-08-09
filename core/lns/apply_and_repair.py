from __future__ import annotations

import contextlib
import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from core.lns.acceptance_criterion import (
    AcceptanceCriterion,
    GreedyAcceptance,
    SimulatedAnnealingAcceptance,
    UnconditionalAcceptance,
)
from core.lns.operators.destroy.selection import DestroySelection, NodeSelection
from core.lns.operators.repair.base import RepairOperator
from core.lns.utils import best_repair, remove_nodes
from core.problems._numba_common import ACCEPT_GREEDY, ACCEPT_SA, ACCEPT_UNCONDITIONAL
from core.problems.solution import BaseSolution

if TYPE_CHECKING:
    from core.lns.config import SearchParams


class ApplyAndRepair[S: BaseSolution, C: DestroySelection](ABC):
    """Applies a :class:`DestroySelection` to a solution and repairs it into candidates.

    ``parallel`` applies every rollout independently from the incumbent, returning a
    ``(*B, K)`` candidate solution scored by objective without acceptance (training and
    the parallel search step). ``sequential`` threads the rollouts one after another
    through the incumbent with acceptance, returning a ``(*B,)`` solution.

    Concrete numba-backed subclasses register under ``(problem_name, repair_name)`` and
    declare a ``selection_type``; ``create`` resolves them. This is the injection point for
    multiple repairs per problem.
    """

    selection_type: ClassVar[type[DestroySelection]]
    problem_name: ClassVar[str | None] = None
    repair_name: ClassVar[str | None] = None
    _registry: ClassVar[dict[tuple[str, str], type[ApplyAndRepair]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        problem_name = cls.__dict__.get("problem_name")
        repair_name = cls.__dict__.get("repair_name")
        if problem_name is None or repair_name is None:
            return
        key = (problem_name, repair_name)
        if key in ApplyAndRepair._registry:
            raise ValueError(
                f"Cannot register {cls.__name__} under {key!r}: "
                f"already registered to {ApplyAndRepair._registry[key].__name__}."
            )
        ApplyAndRepair._registry[key] = cls

    @abstractmethod
    def parallel(self, solution: S, selection: C) -> S:
        """
        Apply every rollout independently starting from the ``solution``.
        Returns ``(*B, K)`` candidate solutions.
        """
        ...

    @abstractmethod
    def sequential(self, solution: S, selection: C, accept: AcceptanceCriterion) -> S:
        """
        Apply the rollouts sequentially starting from the ``solution`` with acceptance.
        Returns a final solution with ``(*B,)``.
        """
        ...

    @classmethod
    def create(cls, problem_name: str, repair_name: str, **kwargs: object) -> ApplyAndRepair:
        # Importing the problem's apply_repair module registers its ApplyAndRepair(s).
        with contextlib.suppress(ModuleNotFoundError):
            importlib.import_module(f"core.problems.{problem_name}.lns.apply_repair")
        subclass = ApplyAndRepair._registry.get((problem_name, repair_name))
        if subclass is None:
            raise ValueError(
                f"No numba ApplyAndRepair registered for problem {problem_name!r} "
                f"repair {repair_name!r}."
            )
        return subclass(**kwargs)


class TorchApplyAndRepair[S: BaseSolution](ApplyAndRepair[S, NodeSelection]):
    selection_type = NodeSelection

    def __init__(self, *, repair: RepairOperator, n_repairs: int) -> None:
        self._repair = repair
        self._n_repairs = n_repairs

    def parallel(self, solution: S, selection: NodeSelection) -> S:
        deconstructed = remove_nodes(solution, selection.nodes, stitch=self._repair.requires_stitch)
        return best_repair(self._repair, deconstructed, self._n_repairs)

    def sequential(self, solution: S, selection: NodeSelection, accept: AcceptanceCriterion) -> S:
        nodes = selection.nodes  # [*B, R, M]
        num_rollouts = nodes.shape[-2]
        current = solution
        for r in range(num_rollouts):
            deconstructed = remove_nodes(
                current, nodes[..., r, :], stitch=self._repair.requires_stitch
            )
            candidate = best_repair(self._repair, deconstructed, self._n_repairs)
            mask = accept(current, candidate)
            current = candidate.where(mask, current)
        return current


class NumbaApplyAndRepair[S: BaseSolution](ApplyAndRepair[S, NodeSelection], ABC):
    selection_type = NodeSelection
    supports_affected_routes_only: ClassVar[bool] = False

    def __init__(
        self,
        *,
        n_repairs: int,
        first_repair_uses_selection_order: bool = False,
        insert_in_affected_routes_only: bool = False,
        seed: int = 0,
    ) -> None:
        if insert_in_affected_routes_only and not self.supports_affected_routes_only:
            raise NotImplementedError(
                f"insert_in_affected_routes_only is not supported by {type(self).__name__}."
            )
        self._n_repairs = n_repairs
        self._first_repair_uses_order = first_repair_uses_selection_order
        self._insert_in_affected_routes_only = insert_in_affected_routes_only
        self._seed = seed

    def parallel(self, solution: S, selection: NodeSelection) -> S:
        out = self._apply_parallel(
            solution.to_numpy(), solution.instance.to_numpy(), selection.to_numpy()
        )
        num_rollouts = selection.nodes.shape[-2]
        # TODO: Is this expand necessary?
        # Expand a view (not the N^2 distance matrix) so the (B, K) candidate can score itself.
        instance = solution.instance.reshape(*solution.batch_size, 1).expand(
            *solution.batch_size, num_rollouts
        )
        return type(solution).from_numpy(
            instance,
            out,
            batch_size=torch.Size([*solution.batch_size, num_rollouts]),
            device=solution.device,
        )

    def sequential(self, solution: S, selection: NodeSelection, accept: AcceptanceCriterion) -> S:
        accept_kind, temperature = acceptance_to_numba(accept)
        out = self._apply_sequential(
            solution.to_numpy(),
            solution.instance.to_numpy(),
            selection.to_numpy(),
            accept_kind,
            temperature,
        )
        return type(solution).from_numpy(
            solution.instance, out, batch_size=solution.batch_size, device=solution.device
        )

    @abstractmethod
    def _apply_parallel(self, solution_arrays, instance_arrays, nodes):  # noqa: ANN001, ANN201
        """Call the problem's parallel numba function; return the ``(G, K, ...)`` output SolutionArrays."""
        ...

    @abstractmethod
    def _apply_sequential(  # noqa: ANN001, ANN201
        self, solution_arrays, instance_arrays, nodes, accept_kind, temperature
    ):
        """Call the problem's sequential numba function; return the ``(G, ...)`` output SolutionArrays."""
        ...


def acceptance_to_numba(accept: AcceptanceCriterion) -> tuple[int, float]:
    """Map an acceptance criterion to the ``(kind, temperature)`` the numba kernel expects."""
    if isinstance(accept, GreedyAcceptance):
        return ACCEPT_GREEDY, 0.0
    if isinstance(accept, UnconditionalAcceptance):
        return ACCEPT_UNCONDITIONAL, 0.0
    if isinstance(accept, SimulatedAnnealingAcceptance):
        return ACCEPT_SA, float(accept.temperature)
    raise NotImplementedError(
        "Numba ApplyAndRepair supports GreedyAcceptance, UnconditionalAcceptance and "
        f"SimulatedAnnealingAcceptance, got {type(accept)}"
    )


def build_apply_and_repair(
    search: SearchParams,
    *,
    problem_name: str,
    backend: Literal["numba", "torch"],
    seed: int = 0,
) -> ApplyAndRepair:
    """Resolve the apply-and-repair for a problem: numba from the registry, torch by wrapping."""
    match backend:
        case "numba":
            return ApplyAndRepair.create(
                problem_name=problem_name,
                repair_name=search.repair_type,
                n_repairs=search.n_repairs,
                first_repair_uses_selection_order=search.first_repair_uses_rollout_order,
                insert_in_affected_routes_only=search.insert_in_affected_routes_only,
                seed=seed,
            )
        case "torch":
            repair = RepairOperator.create(
                problem_name=problem_name,
                repair_type=search.repair_type,
                deconstruction_size=search.deconstruction_size,
            )
            return TorchApplyAndRepair(repair=repair, n_repairs=search.n_repairs)
        case _:
            raise ValueError(f"Unknown apply-and-repair backend: {backend!r}")
