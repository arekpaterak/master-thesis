from abc import ABC, abstractmethod

import torch

from core.problems.tsp.solution import TSPBatchSolution


class DestroyOperator(ABC):
    @abstractmethod
    def destroy(self, solution: TSPBatchSolution) -> TSPBatchSolution: ...

    def __call__(self, solution: TSPBatchSolution) -> TSPBatchSolution:
        return self.destroy(solution)


class RandomSegmentDestroy(DestroyOperator):
    def __init__(self, segment_size: int, *, seed: int | None = None) -> None:
        self.segment_size = segment_size

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def destroy(self, solution: TSPBatchSolution) -> TSPBatchSolution:
        was_batched = False
        if len(solution.batch_size) == 0:
            solution = solution.unsqueeze(0)
            was_batched = True
        B = solution.batch_size[0]
        N = solution.num_locations
        device = solution.tour.device

        new_tour = solution.tour.clone()

        # 1. Choose starting position (0 to n-1) for each batch
        start_indices = torch.randint(0, N, (B,), device=device)

        # 2. Create an offset tensor for the segment [0, 1, ..., k-1]
        offsets = torch.arange(self.segment_size, device=device).repeat(B, 1)

        # 3. Calculate all positions to remove with overflow/wrap-around
        positions_to_remove_at = (start_indices.unsqueeze(1) + offsets) % N

        # 4. Batch-apply the destruction
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        new_tour[batch_idx, positions_to_remove_at] = -1

        if was_batched:
            solution = solution.squeeze(0)
            new_tour = new_tour.squeeze(0)

        return TSPBatchSolution(
            instance=solution.instance,
            tour=new_tour,
            batch_size=solution.batch_size,
        )
