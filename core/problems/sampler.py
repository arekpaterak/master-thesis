from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from core.problems.instance import BaseProblemInstance


class Sampler[T: BaseProblemInstance](ABC):
    @abstractmethod
    def __call__(self, batch_size: int) -> T: ...

    def reset(self) -> None:
        """Rewind any internal state so the next pass starts from the beginning.

        Seeded random samplers restore their generator to the initial seed state,
        producing the same instances on every pass. Dataset-backed samplers rewind
        their read position. Unseeded random samplers keep this a no-op.
        """

    @classmethod
    def for_size(cls, size: int, *, seed: int | None = None) -> Self:
        raise NotImplementedError(f"{cls.__name__} does not support construction from size alone.")
