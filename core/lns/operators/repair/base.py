from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from core.problems.solution import BaseSolution


class RepairOperator[S: BaseSolution](ABC):
    requires_stitch: bool = False

    # Registry keys. A subclass that sets its own 'repair_type' is registered;
    # 'problem_name' may stay None to mean the operator applies to any problem.
    repair_type: ClassVar[str | None] = None
    problem_name: ClassVar[str | None] = None
    _registry: ClassVar[dict[tuple[str, str | None], type[RepairOperator]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        repair_type = cls.__dict__.get("repair_type")
        if repair_type is None:
            return
        key = (repair_type, cls.__dict__.get("problem_name"))
        if key in RepairOperator._registry:
            raise ValueError(
                f"Cannot register {cls.__name__} under {key!r}: "
                f"already registered to {RepairOperator._registry[key].__name__}."
            )
        RepairOperator._registry[key] = cls

    @abstractmethod
    def __call__(self, deconstructed_solution: S) -> S: ...

    @classmethod
    def build(cls, *, deconstruction_size: int | None = None) -> RepairOperator:
        return cls(deconstruction_size=deconstruction_size)

    @classmethod
    def create(
        cls, problem_name: str, repair_type: str, *, deconstruction_size: int | None = None
    ) -> RepairOperator:
        registry = RepairOperator._registry
        # Prefer a problem-specific operator, falling back to a problem-agnostic one.
        subclass = registry.get((repair_type, problem_name)) or registry.get((repair_type, None))
        if subclass is None:
            if not any(registered_type == repair_type for registered_type, _ in registry):
                raise ValueError(f"Unsupported repair type: {repair_type!r}")
            raise ValueError(
                f"Repair type {repair_type!r} is not implemented for problem {problem_name!r}."
            )
        return subclass.build(deconstruction_size=deconstruction_size)


# TODO: Remove as this might not produce complete solution
class NoopRepair[S: BaseSolution](RepairOperator[S]):
    repair_type = "noop"

    def __call__(self, deconstructed_solution: S) -> S:
        return deconstructed_solution

    @classmethod
    def build(cls, *, deconstruction_size: int | None = None) -> NoopRepair:
        return cls()
