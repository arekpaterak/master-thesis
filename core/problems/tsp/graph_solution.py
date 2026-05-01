from __future__ import annotations

import torch

from core.problems.solution import BaseSolution
from core.problems.tsp import TSPInstance


class TSPGraphSolution(BaseSolution):
    _instance: TSPInstance
    _next: torch.Tensor
    _prev: torch.Tensor

    UNASSIGNED = -1

    @classmethod
    def create(
        cls, instance: TSPInstance, sequence: torch.Tensor | None = None
    ) -> TSPGraphSolution:
        if sequence is None:
            return cls(
                _instance=instance,
                _next=torch.full(
                    (*instance.batch_size, instance.num_nodes),
                    cls.UNASSIGNED,
                    dtype=torch.long,
                    device=instance.device,
                ),
                _prev=torch.full(
                    (*instance.batch_size, instance.num_nodes),
                    cls.UNASSIGNED,
                    dtype=torch.long,
                    device=instance.device,
                ),
            )

        assert not (sequence == cls.UNASSIGNED).any(), "Expected complete tours"

        b_size = instance.batch_size
        n = instance.num_nodes
        next_nodes = torch.full(
            (*b_size, n), cls.UNASSIGNED, dtype=torch.long, device=sequence.device
        )
        prev_nodes = torch.full(
            (*b_size, n), cls.UNASSIGNED, dtype=torch.long, device=sequence.device
        )

        next_nodes.scatter_(-1, sequence, sequence.roll(-1, dims=-1))
        prev_nodes.scatter_(-1, sequence, sequence.roll(1, dims=-1))

        return cls(_instance=instance, _next=next_nodes, _prev=prev_nodes)

    @property
    def instance(self) -> TSPInstance:
        return self._instance

    @property
    def next_nodes(self) -> torch.Tensor:
        return self._next

    @property
    def prev_nodes(self) -> torch.Tensor:
        return self._prev

    @property
    def is_feasible(self) -> torch.Tensor:
        pass

    @property
    def num_unassigned(self) -> torch.Tensor:
        return (self._next == self.UNASSIGNED).sum(dim=-1)

    def objective(self) -> torch.Tensor:
        pass

    def node_degree(self) -> torch.Tensor:
        pass

    def visualize(self, **kwargs) -> None:
        pass

    @property
    def is_complete(self) -> torch.Tensor:
        pass

    def remove_edges(self, *, to_nodes: torch.Tensor) -> None:
        if to_nodes.shape == self._next.shape[:-1]:
            to_nodes = to_nodes.unsqueeze(-1)
        from_node = self._prev.gather(-1, to_nodes)

        self._next.scatter_(-1, from_node, self.UNASSIGNED)
        self._prev.scatter_(-1, to_nodes, self.UNASSIGNED)
        return

    def add_edges(self, edges: torch.Tensor) -> None:
        from_nodes = edges[..., 0]
        to_nodes = edges[..., 1]

        assert (self._next.gather(-1, from_nodes) == self.UNASSIGNED).all(), (
            "from_node already has a successor"
        )
        assert (self._prev.gather(-1, to_nodes) == self.UNASSIGNED).all(), (
            "to_node already has a predecessor"
        )

        self._next.scatter_(-1, index=from_nodes, src=to_nodes)
        self._prev.scatter_(-1, index=to_nodes, src=from_nodes)
