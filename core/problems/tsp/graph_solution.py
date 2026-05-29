from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.problems.solution import BaseSolution
from core.problems.tsp import TSPInstance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class TSPGraphSolution(BaseSolution):
    _instance: TSPInstance
    _next: torch.Tensor
    _prev: torch.Tensor

    UNASSIGNED = -1

    @classmethod
    def create_from_sequence(
        cls, instance: TSPInstance, sequence: torch.Tensor | None = None
    ) -> TSPGraphSolution:
        device = instance.locations.device

        if sequence is None:
            return cls(
                _instance=instance,
                _next=torch.full(
                    (*instance.batch_size, instance.num_nodes),
                    cls.UNASSIGNED,
                    dtype=torch.long,
                    device=device,
                ),
                _prev=torch.full(
                    (*instance.batch_size, instance.num_nodes),
                    cls.UNASSIGNED,
                    dtype=torch.long,
                    device=device,
                ),
                batch_size=instance.batch_size,
            )

        assert not (sequence == cls.UNASSIGNED).any(), "Expected complete tours"

        b_size = instance.batch_size
        n = instance.num_nodes
        next_nodes = torch.full((*b_size, n), cls.UNASSIGNED, dtype=torch.long, device=device)
        prev_nodes = torch.full((*b_size, n), cls.UNASSIGNED, dtype=torch.long, device=device)

        next_nodes.scatter_(-1, sequence, sequence.roll(-1, dims=-1))
        prev_nodes.scatter_(-1, sequence, sequence.roll(1, dims=-1))

        return cls(_instance=instance, _next=next_nodes, _prev=prev_nodes, batch_size=b_size)

    def __post_init__(self) -> None:
        pass

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
        raise NotImplementedError()

    @property
    def num_unassigned(self) -> torch.Tensor:
        return (self._next == self.UNASSIGNED).sum(dim=-1)

    def objective(self) -> torch.Tensor:
        locations = self._instance.locations
        next_locations = locations.gather(-2, self._next.unsqueeze(-1).expand(*self._next.shape, 2))
        length = torch.linalg.norm(locations - next_locations, dim=-1)
        return length.sum(dim=-1)

    def neighbours_index(self) -> torch.Tensor:
        return torch.stack([self._prev, self._next], dim=-1)

    def node_degree(self) -> torch.Tensor:
        return (self._next != self.UNASSIGNED).long() + (self._prev != self.UNASSIGNED).long()

    def visualize(
        self,
        *,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
        with_problem_size: bool = True,
        with_objective: bool = True,
    ) -> None:
        from .render import render_solution

        render_solution(
            self,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            ax=ax,
            with_problem_size=with_problem_size,
            with_objective=with_objective,
        )

    @property
    def is_complete(self) -> torch.Tensor:
        return self.num_unassigned == 0

    def remove_nodes(self, nodes: torch.Tensor) -> TSPGraphSolution:
        target_batch_shape = nodes.shape[:-1]

        shape_diff = len(target_batch_shape) - len(self.batch_size)
        expanded = (
            self.reshape(*self.batch_size, *[1] * shape_diff).expand(target_batch_shape).clone()
        )

        predecessors = expanded._prev.gather(-1, nodes)
        successors = expanded._next.gather(-1, nodes)
        expanded._next.scatter_(-1, predecessors, self.UNASSIGNED)
        expanded._prev.scatter_(-1, nodes, self.UNASSIGNED)
        expanded._next.scatter_(-1, nodes, self.UNASSIGNED)
        expanded._prev.scatter_(-1, successors, self.UNASSIGNED)

        return expanded

    def remove_edges(self, *, end_nodes: torch.Tensor) -> None:
        if end_nodes.shape == self._next.shape[:-1]:
            end_nodes = end_nodes.unsqueeze(-1)
        start_nodes = self._prev.gather(-1, end_nodes)

        self._next.scatter_(-1, start_nodes, self.UNASSIGNED)
        self._prev.scatter_(-1, end_nodes, self.UNASSIGNED)

    def add_edges(self, edges: torch.Tensor) -> None:
        start_nodes = edges[..., 0]
        end_nodes = edges[..., 1]

        assert (self._next.gather(-1, start_nodes) == self.UNASSIGNED).all(), (
            "from_node already has a successor"
        )
        assert (self._prev.gather(-1, end_nodes) == self.UNASSIGNED).all(), (
            "to_node already has a predecessor"
        )

        self._next.scatter_(-1, index=start_nodes, src=end_nodes)
        self._prev.scatter_(-1, index=end_nodes, src=start_nodes)
