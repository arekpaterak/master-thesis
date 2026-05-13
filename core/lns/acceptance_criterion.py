from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from core.problems.solution import BaseSolution


class AcceptanceCriterion(ABC):
    def accept(self, *, current: BaseSolution, candidate: BaseSolution) -> torch.Tensor:
        """
        Evaluate candidate solutions against current ones.

        Args:
            current: The current batch of solutions.
            candidate: The candidate batch of solutions.

        Returns:
            accept_mask: (batch,) bool tensor where True means the candidate solution is accepted.
        """
        assert current.batch_size == candidate.batch_size, (
            "Expected current and candidates to have the same batch size"
        )
        return self._accept(current=current, candidate=candidate)

    @abstractmethod
    def _accept(self, *, current: BaseSolution, candidate: BaseSolution) -> torch.Tensor: ...

    def __call__(self, *, current: BaseSolution, candidate: BaseSolution) -> torch.Tensor:
        return self.accept(current=current, candidate=candidate)


class GreedyAcceptance(AcceptanceCriterion):
    def _accept(self, *, current: BaseSolution, candidate: BaseSolution) -> torch.Tensor:
        return candidate.objective() < current.objective()
