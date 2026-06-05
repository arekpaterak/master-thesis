import torch

from core.lns.operators.destroy.base import DestroyOperator
from core.problems.tsp.graph_solution import TSPGraphSolution


# Deprecated
class TSPRandomEdgeDestroy(DestroyOperator[TSPGraphSolution]):
    def __init__(self, destruction_size: int, *, seed: int | None = None) -> None:
        if not destruction_size >= 0:
            raise ValueError("Expected non-negative destruction size.")
        self.destruction_size = destruction_size

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(self, solution: TSPGraphSolution) -> TSPGraphSolution:
        num_nodes = solution.instance.num_nodes
        device = solution.device
        batch_size = solution.batch_size

        if self.destruction_size > num_nodes:
            raise ValueError(f"Destruction size {self.destruction_size} exceeds nodes {num_nodes}")

        s = solution.clone()

        to_nodes = (
            torch.rand((*batch_size, num_nodes), device=device, generator=self._generator)
            .topk(self.destruction_size, dim=-1)
            .indices
        )
        print(to_nodes)
        s.remove_edges(end_nodes=to_nodes)
        return s
