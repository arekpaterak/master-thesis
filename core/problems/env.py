from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution

if TYPE_CHECKING:
    from pathlib import Path

    from core.problems.config import BaseProblemConfig
    from core.problems.features import InstanceFeatureExtractor, SolutionFeatureExtractor
    from core.problems.sampler import Sampler
    from core.states.destroy import DestroyState


class ProblemEnv[I: BaseProblemInstance, S: BaseSolution](ABC):
    name: ClassVar[str]
    depot_offset: ClassVar[int]
    """Number of leading non-customer nodes in a (B, N + offset, D) node tensor."""

    instance_cls: ClassVar[type[BaseProblemInstance]]
    solution_cls: ClassVar[type[BaseSolution]]

    _registry: ClassVar[dict[str, type[ProblemEnv]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Registration is opt-in: a subclass without its own 'name' is not registered.
        name = cls.__dict__.get("name")
        if name is None:
            return
        if name in ProblemEnv._registry:
            raise ValueError(
                f"Cannot register {cls.__name__} under {name!r}: "
                f"already registered to {ProblemEnv._registry[name].__name__}."
            )
        ProblemEnv._registry[name] = cls

    @classmethod
    def resolve(cls, name: str) -> type[ProblemEnv]:
        try:
            return ProblemEnv._registry[name]
        except KeyError:
            raise ValueError(f"Unsupported problem name: {name!r}") from None

    # --- construction ---
    @abstractmethod
    def sampler(self, config: BaseProblemConfig, *, seed: int | None = None) -> Sampler[I]: ...

    @abstractmethod
    def load_dataset(self, path: str | Path) -> I: ...

    @abstractmethod
    def initial_solution(self, instance: I) -> S: ...

    # --- search ---
    @abstractmethod
    def destroy_state_cls(self) -> type[DestroyState]: ...

    # --- encoding ---
    @abstractmethod
    def instance_features(
        self, *, with_positional: bool
    ) -> InstanceFeatureExtractor[I, object]: ...

    @abstractmethod
    def solution_features(
        self, *, with_positional: bool
    ) -> SolutionFeatureExtractor[S, object]: ...
