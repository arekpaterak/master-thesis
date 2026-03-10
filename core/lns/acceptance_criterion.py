from abc import ABC, abstractmethod

import torch

from core.problems.tsp.solution import TSPBatchSolution


class AcceptanceCriterion(ABC):
    def accept(self, current: TSPBatchSolution, candidates: TSPBatchSolution) -> torch.Tensor:
        """
        Args:
            current: The current batch of solutions.
            candidates: The candidate batch of solutions.

        Returns:
            accept_mask: (batch,) bool tensor where True means the candidate solution is accepted.
        """
        assert current.batch_size == candidates.batch_size, (
            "Expected current and candidates to have the same batch size"
        )
        return self._accept(current, candidates)

    @abstractmethod
    def _accept(self, current: TSPBatchSolution, candidates: TSPBatchSolution) -> torch.Tensor: ...

    def __call__(self, current: TSPBatchSolution, candidates: TSPBatchSolution) -> torch.Tensor:
        return self.accept(current, candidates)


class GreedyAcceptance(AcceptanceCriterion):
    def _accept(self, current: TSPBatchSolution, candidates: TSPBatchSolution) -> torch.Tensor:
        return candidates.tour_lengths < current.tour_lengths
