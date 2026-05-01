import torch

from core.lns.operators.destroy.base import DestroyOperator
from core.problems.cvrp import CVRPSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.solution import TSPSolution


class RandomNodeDestroy[S: BaseSolution](DestroyOperator[S]):
    def __init__(self, destruction_size: int, *, seed: int | None = None) -> None:
        if not destruction_size >= 0:
            raise ValueError("Expected non-negative destruction size.")
        self.destruction_size = destruction_size

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(self, solution: S) -> S:
        num_nodes = solution.instance.num_nodes
        device = solution.device
        batch_size = solution.batch_size

        match solution:
            case TSPSolution() as s:
                if self.destruction_size > num_nodes:
                    raise ValueError(
                        f"Destruction size {self.destruction_size} exceeds nodes {num_nodes}"
                    )

                to_remove = (
                    torch.rand((*batch_size, num_nodes), device=device, generator=self._generator)
                    .topk(self.destruction_size, dim=-1)
                    .indices
                )
                return s.remove_nodes(to_remove)  # type: ignore

            case CVRPSolution() as s:
                num_customers = s.instance.num_customers
                if self.destruction_size > num_customers:
                    raise ValueError(
                        f"Destruction size {self.destruction_size} exceeds customers {num_customers}"
                    )

                to_remove = (
                    torch.rand(
                        (*batch_size, num_customers), device=device, generator=self._generator
                    )
                    .topk(self.destruction_size, dim=-1)
                    .indices
                    + 1
                )
                return s.remove_customers(to_remove)  # type: ignore

            case _:
                raise NotImplementedError(f"Operator not implemented for {type(solution)}")
