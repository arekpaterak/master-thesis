import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.solution import BaseSolution
from core.problems.tsp.solution import TSPSolution


class RandomRepair[S: BaseSolution](RepairOperator[S]):
    def __init__(self, *, seed: int | None = None) -> None:
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(self, deconstructed_solution: S) -> S:
        match deconstructed_solution:
            case TSPSolution() as s:
                tour = s.tour.clone()
                unassigned_mask = s.unassigned_mask
                unassigned_nodes = s.unassigned_nodes()

                b_size = s.batch_size
                k = unassigned_nodes.shape[-1]
                if k > 0:
                    rand = torch.rand(
                        *b_size, k, device=unassigned_nodes.device, generator=self._generator
                    )
                    order = rand.argsort(dim=-1)
                    shuffled_nodes = unassigned_nodes.gather(dim=-1, index=order)
                    tour[unassigned_mask] = shuffled_nodes.flatten()

                return TSPSolution.create(instance=s.instance, tour=tour)

            case _:
                raise NotImplementedError(
                    f"Operator not implemented for {type(deconstructed_solution)}"
                )
