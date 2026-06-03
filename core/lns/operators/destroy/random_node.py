import torch

from core.lns.operators.destroy.base import DestroyOperator
from core.problems.cvrp import CVRPGraphSolution, CVRPSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.solution import TSPSolution


class RandomNodeDestroy[S: BaseSolution](DestroyOperator[S]):
    def __init__(
        self,
        destruction_size: int,
        *,
        seed: int | None = None,
        stitch: bool = False,
    ) -> None:
        if not destruction_size >= 0:
            raise ValueError("Expected non-negative destruction size.")
        self.destruction_size = destruction_size
        self.stitch = stitch

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(self, solution: S) -> S:
        # TODO: Refactor to be problem-agnostic
        num_nodes = solution.instance.num_nodes
        device = solution.instance.locations.device
        batch_size = solution.batch_size

        match solution:
            case TSPSolution() | TSPGraphSolution() as s:
                if self.destruction_size > num_nodes:
                    raise ValueError(
                        f"Destruction size {self.destruction_size} exceeds number of nodes {num_nodes}"
                    )

                nodes_to_remove = (
                    torch.rand((*batch_size, num_nodes), generator=self._generator)
                    .topk(self.destruction_size, dim=-1)
                    .indices.to(device=device)
                )
                return s.remove_nodes(nodes_to_remove)

            case CVRPSolution() | CVRPGraphSolution() as s:
                num_customers = s.instance.num_customers
                if self.destruction_size > num_customers:
                    raise ValueError(
                        f"Destruction size {self.destruction_size} exceeds number of customers {num_customers}"
                    )

                nodes_to_remove = (
                    torch.rand((*batch_size, num_customers), generator=self._generator)
                    .topk(self.destruction_size, dim=-1)
                    .indices.to(device=device)
                    + 1  # offset as customers are indexed from 1 to num_customers, while depot is indexed as 0
                )
                if isinstance(s, CVRPGraphSolution):
                    return s.remove_nodes(nodes_to_remove, stitch=self.stitch)
                return s.remove_nodes(nodes_to_remove)

            case _:
                raise NotImplementedError(f"Operator not implemented for {type(solution)}")
