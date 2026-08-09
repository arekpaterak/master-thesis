from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar

from core.lns.operators.destroy.selection import DestroySelection
from core.problems.solution import BaseSolution


class DestroyOperator[S: BaseSolution, C: DestroySelection](ABC):
    """Selects what to remove from a solution.

    Destroy never returns a destructed solution: it emits a :class:`DestroySelection`
    that an ``ApplyAndRepair`` of the matching ``selection_type`` then applies and repairs.
    """

    selection_type: ClassVar[type[DestroySelection]]

    @abstractmethod
    def select(self, solution: S, num_rollouts: int) -> C: ...

    @contextmanager
    def search_session(self, solution: S) -> Iterator[None]:
        yield
