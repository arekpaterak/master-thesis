from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

from core.problems.instance import BaseProblemInstance

if TYPE_CHECKING:
    from core.problems.config import BaseProblemConfig


class Sampler[T: BaseProblemInstance](ABC):
    problem_name: ClassVar[str]
    _registry: ClassVar[dict[str, type[Sampler]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Registration is opt-in: a subclass without its own 'problem_name'
        # (e.g. an experimental sampler outside the factory) is simply not registered.
        name = cls.__dict__.get("problem_name")
        if name is None:
            return
        if name in Sampler._registry:
            raise ValueError(
                f"Cannot register {cls.__name__} under {name!r}: "
                f"already registered to {Sampler._registry[name].__name__}."
            )
        Sampler._registry[name] = cls

    @abstractmethod
    def __call__(self, batch_size: int) -> T: ...

    def reset(self) -> None:
        """Rewind any internal position so the next pass starts from the beginning.

        Random samplers are stateless and keep this a no-op; dataset-backed
        samplers override it to re-serve the same fixed instances each pass.
        """

    @classmethod
    def for_size(cls, size: int, *, seed: int | None = None) -> Self:
        """Build a sampler for a problem of the given size.

        Implemented by samplers that participate in the factory (those with a
        ``problem_name``); ``Sampler.create`` dispatches here.
        """
        raise NotImplementedError(f"{cls.__name__} does not support construction from size alone.")

    @classmethod
    def create(cls, problem_name: str, size: int, *, seed: int | None = None) -> Sampler:
        """Build a sampler from a problem name and size (lightweight convenience)."""
        return cls._resolve(problem_name).for_size(size, seed=seed)

    @classmethod
    def from_config(cls, config: BaseProblemConfig, *, seed: int | None = None) -> Sampler:
        """Build a sampler from a problem config, honoring its problem-specific fields."""
        return cls._resolve(config.name)(config.to_problem(), seed=seed)

    @classmethod
    def _resolve(cls, problem_name: str) -> type[Sampler]:
        try:
            return Sampler._registry[problem_name]
        except KeyError:
            raise ValueError(f"Unsupported problem name: {problem_name!r}") from None
