import torch

from core.lns.operators.destroy.base import DestroyOperator
from core.problems.tsp.solution import TSPSolution


class RandomSegmentDestroy(DestroyOperator[TSPSolution]):
    def __init__(self, segment_size: int, *, seed: int | None = None) -> None:
        if not segment_size >= 0:
            raise ValueError("Expected non-negative segment size.")
        self.segment_size = segment_size

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def destroy(self, solution: TSPSolution) -> TSPSolution:
        num_nodes = solution.instance.num_nodes
        if self.segment_size > num_nodes:
            raise ValueError("Expected problems bigger than destruction size.")

        was_unbatched = len(solution.batch_size) == 0
        tour = solution.tour.unsqueeze(0) if was_unbatched else solution.tour
        B = tour.shape[0]
        N = solution.instance.num_nodes
        device = tour.device

        new_tour = tour.clone()

        # 1. Choose starting position (0 to n-1) for each batch
        start_indices = torch.randint(0, N, (B,), device=device, generator=self._generator)

        # 2. Create an offset tensor for the segment [0, 1, ..., k-1]
        offsets = torch.arange(self.segment_size, device=device).repeat(B, 1)
        # TODO: Remove .repeat()?

        # 3. Calculate all positions to remove with overflow/wrap-around
        positions_to_remove_at = (start_indices.unsqueeze(1) + offsets) % N

        # 4. Batch-apply the destruction
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        new_tour[batch_idx, positions_to_remove_at] = -1

        if was_unbatched:
            new_tour = new_tour.squeeze(0)

        return TSPSolution.create(
            instance=solution.instance,
            tour=new_tour,
        )
