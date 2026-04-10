from abc import ABC, abstractmethod

import torch

from core.problems.tsp.solution import TSPSolution


class AcceptanceCriterion(ABC):
    def accept(self, *, current: TSPSolution, candidates: TSPSolution) -> torch.Tensor:
        """
        Evaluate candidate solutions against current ones.

        Args:
            current: The current batch of solutions.
            candidates: The candidate batch of solutions.

        Returns:
            accept_mask: (batch,) bool tensor where True means the candidate solution is accepted.
        """
        assert current.batch_size == candidates.batch_size, (
            "Expected current and candidates to have the same batch size"
        )
        return self._accept(current=current, candidates=candidates)

    @abstractmethod
    def _accept(self, *, current: TSPSolution, candidates: TSPSolution) -> torch.Tensor: ...

    def __call__(self, *, current: TSPSolution, candidates: TSPSolution) -> torch.Tensor:
        return self.accept(current=current, candidates=candidates)


class GreedyAcceptance(AcceptanceCriterion):
    def _accept(self, *, current: TSPSolution, candidates: TSPSolution) -> torch.Tensor:
        return candidates.objective() < current.objective()
