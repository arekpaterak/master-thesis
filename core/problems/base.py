from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from core.problems.instance import BaseProblemInstance
from core.problems.names import ProblemName
from core.problems.sampler import Sampler
from core.problems.solution import BaseSolution
from core.resolve import resolve


@dataclass(frozen=True)
class Problem[I: BaseProblemInstance, S: BaseSolution]:
    name: ProblemName
    sampler: type[Sampler[I]]
    load_dataset: Callable[[str | Path], I]
    initial_solution: Callable[[I], S]
    constructions: Mapping[str, Callable[[I], S]] = field(default_factory=dict)

    def construction(self, name: str) -> Callable[[I], S]:
        return resolve(self.constructions, name, kind="construction", problem=self.name)
