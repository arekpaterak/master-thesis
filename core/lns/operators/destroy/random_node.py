import torch

from core.lns.operators.destroy.base import DestroyOperator
from core.lns.operators.destroy.selection import NodeSelection
from core.problems.solution import BaseSolution


class RandomNodeDestroy[S: BaseSolution](DestroyOperator[S, NodeSelection]):
    selection_type = NodeSelection

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

    def select(self, solution: S, num_rollouts: int) -> NodeSelection:
        """Return ``[*B, R, M]`` node indices sampled uniformly without replacement."""
        instance = solution.instance
        num_candidates = instance.num_nodes - instance.num_fixed_nodes
        if self.destruction_size > num_candidates:
            raise ValueError(
                f"Destruction size {self.destruction_size} exceeds the {num_candidates} "
                "removable nodes."
            )
        actions = (
            torch.rand(
                (*solution.batch_size, num_rollouts, num_candidates), generator=self._generator
            )
            .topk(self.destruction_size, dim=-1)
            .indices.to(device=instance.locations.device)
            + instance.num_fixed_nodes
        )
        return NodeSelection(actions)

    def __call__(self, solution: S) -> S:
        nodes = self.select(solution, num_rollouts=1).nodes.squeeze(-2)
        return solution.remove_nodes(nodes, stitch=self.stitch)
