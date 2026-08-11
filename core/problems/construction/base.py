from __future__ import annotations

import contextlib
import importlib
from abc import ABC, abstractmethod
from typing import ClassVar

from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class ConstructionHeuristic[I: BaseProblemInstance, S: BaseSolution](ABC):
    """Builds a feasible solution from an instance: ``(instance) -> solution``.

    A subclass that sets its own ``construction_name`` self-registers under
    ``(construction_name, problem_name)``.
    ``problem_name`` may stay None to mean the heuristic applies to any problem.
    """

    construction_name: ClassVar[str | None] = None
    problem_name: ClassVar[str | None] = None
    _registry: ClassVar[dict[tuple[str, str | None], type[ConstructionHeuristic]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        construction_name = cls.__dict__.get("construction_name")
        if construction_name is None:
            return
        key = (construction_name, cls.__dict__.get("problem_name"))
        if key in ConstructionHeuristic._registry:
            raise ValueError(
                f"Cannot register {cls.__name__} under {key!r}: "
                f"already registered to {ConstructionHeuristic._registry[key].__name__}."
            )
        ConstructionHeuristic._registry[key] = cls

    @abstractmethod
    def __call__(self, instance: I) -> S: ...

    @classmethod
    def create(cls, problem_name: str, construction_name: str) -> ConstructionHeuristic:
        # Importing the problem's construction module registers its ConstructionHeuristics.
        with contextlib.suppress(ModuleNotFoundError):
            importlib.import_module(f"core.problems.{problem_name}.construction")
        registry = ConstructionHeuristic._registry
        # Prefer a problem-specific heuristic, falling back to a problem-agnostic one.
        subclass = registry.get((construction_name, problem_name)) or registry.get(
            (construction_name, None)
        )
        if subclass is None:
            if not any(registered == construction_name for registered, _ in registry):
                raise ValueError(f"Unsupported construction name: {construction_name!r}")
            raise ValueError(
                f"Construction name {construction_name!r} is not implemented "
                f"for problem {problem_name!r}."
            )
        return subclass()
