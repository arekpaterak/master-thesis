import torch

from core.lns.operators.destroy.base import NodeDestroyOperator
from core.problems.cmst.solution import CMSTSolution
from core.problems.cvrp import CVRPGraphSolution, CVRPSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.solution import TSPSolution


class RandomNodeDestroy[S: BaseSolution](NodeDestroyOperator[S]):
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

    def select_nodes(self, solution: S, num_rollouts: int) -> torch.Tensor:
        """Return ``[*B, R, M]`` node indices sampled uniformly without replacement."""
        num_nodes = solution.instance.num_nodes
        device = solution.instance.locations.device
        batch_size = solution.batch_size

        match solution:
            case TSPSolution() | TSPGraphSolution():
                if self.destruction_size > num_nodes:
                    raise ValueError(
                        f"Destruction size {self.destruction_size} exceeds number of nodes {num_nodes}"
                    )
                return (
                    torch.rand((*batch_size, num_rollouts, num_nodes), generator=self._generator)
                    .topk(self.destruction_size, dim=-1)
                    .indices.to(device=device)
                )

            case CVRPSolution() | CVRPGraphSolution() as s:
                num_customers = s.instance.num_customers
                if self.destruction_size > num_customers:
                    raise ValueError(
                        f"Destruction size {self.destruction_size} exceeds number of customers {num_customers}"
                    )
                return (
                    torch.rand(
                        (*batch_size, num_rollouts, num_customers), generator=self._generator
                    )
                    .topk(self.destruction_size, dim=-1)
                    .indices.to(device=device)
                    + 1  # offset as customers are indexed from 1 to num_customers, while depot is indexed as 0
                )

            case CMSTSolution() as s:
                num_terminals = s.instance.num_terminals
                if self.destruction_size > num_terminals:
                    raise ValueError(
                        f"Destruction size {self.destruction_size} exceeds number of terminals {num_terminals}"
                    )
                return (
                    torch.rand(
                        (*batch_size, num_rollouts, num_terminals), generator=self._generator
                    )
                    .topk(self.destruction_size, dim=-1)
                    .indices.to(device=device)
                    + 1  # offset as terminals are indexed from 1 to num_terminals, while the root is indexed as 0
                )

            case _:
                raise NotImplementedError(f"Operator not implemented for {type(solution)}")

    def __call__(self, solution: S) -> S:
        nodes = self.select_nodes(solution, num_rollouts=1).squeeze(-2)
        match solution:
            case TSPGraphSolution() as s:
                return s.remove_nodes(nodes, stitch=self.stitch)
            case TSPSolution() as s:
                return s.remove_nodes(nodes)
            case CVRPSolution() | CVRPGraphSolution() as s:
                if isinstance(s, CVRPGraphSolution):
                    return s.remove_nodes(nodes, stitch=self.stitch)
                return s.remove_nodes(nodes)
            case _:
                raise NotImplementedError(f"Operator not implemented for {type(solution)}")
