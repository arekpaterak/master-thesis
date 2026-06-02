from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal, Self

import torch
from pydantic import BaseModel, Field

from core.constants import EPS
from core.problems.solution import BaseSolution


class AcceptanceCriterion(ABC):
    @abstractmethod
    def _accept(self, *, current: BaseSolution, candidate: BaseSolution) -> torch.Tensor: ...

    def __call__(self, current: BaseSolution, candidate: BaseSolution) -> torch.Tensor:
        assert current.batch_size == candidate.batch_size, (
            "Expected current and candidates to have the same batch size"
        )
        return self._accept(current=current, candidate=candidate)

    def reset(self) -> None:
        pass

    def step(self) -> None:
        pass


class GreedyAcceptance(AcceptanceCriterion):
    """Accepts a candidate only if it strictly improves on the current solution."""

    def _accept(self, *, current: BaseSolution, candidate: BaseSolution) -> torch.Tensor:
        return candidate.objective() < current.objective()


class SimulatedAnnealingAcceptance(AcceptanceCriterion):
    """Simulated annealing acceptance with exponential temperature decay.

    Temperature is constant within a single improvement step (across all K rollouts)
    and decays once per outer iteration via step().
    Call reset() to restart the schedule.

    Always accepts improved solutions. Accepts worse solutions with probability
    exp(-delta / temperature).
    """

    def __init__(self, *, temperature: float, decay: float) -> None:
        self._temperature_start = temperature
        self._decay = decay
        self._temperature = temperature

    @classmethod
    def from_range(cls, *, temperature_start: float, temperature_min: float, n_steps: int) -> Self:
        """Construct with exponential decay from temperature_start to temperature_min.

        Args:
            temperature_start: The temperature at the first iteration.
            temperature_min: The target temperature after n_steps.
            n_steps: The number of decay steps.

        Returns:
            A criterion whose temperature decays exponentially over n_steps.
        """
        assert n_steps > 0, "Expected n_steps > 0"
        decay = (temperature_min / temperature_start) ** (1 / n_steps)
        return cls(temperature=temperature_start, decay=decay)

    @property
    def temperature(self) -> float:
        return self._temperature

    def reset(self) -> None:
        self._temperature = self._temperature_start

    def step(self) -> None:
        self._temperature *= self._decay

    def _accept(
        self,
        *,
        current: BaseSolution,
        candidate: BaseSolution,
    ) -> torch.Tensor:
        delta = candidate.objective() - current.objective()
        prob = torch.exp(-delta.clamp(min=0) / self._temperature)
        return (delta < EPS) | (torch.rand_like(delta) < prob)


class GreedyAcceptanceConfig(BaseModel):
    type: Literal["greedy"] = "greedy"


class SimulatedAnnealingAcceptanceConfig(BaseModel):
    type: Literal["sa"] = "sa"
    temperature_start: float = Field(default=0.1, description="SA initial temperature.")
    temperature_min: float = Field(default=0.001, description="SA minimum temperature.")


AcceptanceCriterionConfig = Annotated[
    GreedyAcceptanceConfig | SimulatedAnnealingAcceptanceConfig,
    Field(discriminator="type"),
]


def build_criterion(config: AcceptanceCriterionConfig, *, n_steps: int) -> AcceptanceCriterion:
    match config:
        case GreedyAcceptanceConfig():
            return GreedyAcceptance()
        case SimulatedAnnealingAcceptanceConfig():
            return SimulatedAnnealingAcceptance.from_range(
                temperature_start=config.temperature_start,
                temperature_min=config.temperature_min,
                n_steps=n_steps,
            )
