from abc import ABC, abstractmethod

import torch

from core.problems.tsp.solution import TSPBatchSolution


class DestroyOperator(ABC):
    @abstractmethod
    def destroy(self, solutions: TSPBatchSolution) -> TSPBatchSolution: ...

    def __call__(self, solutions: TSPBatchSolution) -> TSPBatchSolution:
        return self.destroy(solutions)


class RandomSegmentDestroy(DestroyOperator):
    def __init__(self, segment_size: int, *, seed: int | None = None) -> None:
        self.segment_size = segment_size

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def destroy(self, solutions: TSPBatchSolution) -> TSPBatchSolution:
        batch_size, n = solutions.tour.shape
        device = solutions.tour.device

        new_tour = solutions.tour.clone()

        # 1. Choose starting position (0 to n-1) for each batch
        start_indices = torch.randint(0, n, (batch_size,), device=device)

        # 2. Create an offset tensor for the segment [0, 1, ..., k-1]
        offsets = torch.arange(self.segment_size, device=device).repeat(batch_size, 1)

        # 3. Calculate all positions to remove with overflow/wrap-around
        positions_to_remove_at = (start_indices.unsqueeze(1) + offsets) % n

        # 4. Batch-apply the destruction
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        new_tour[batch_idx, positions_to_remove_at] = -1

        return TSPBatchSolution(
            instance=solutions.instance,
            tour=new_tour,
        )
