from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from core.constants import EPS
from core.problems.solution import BaseSolution
from core.problems.tsp import TSPInstance
from core.problems.tsp.arrays import TSPSolutionArrays

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

    def to_numpy(self) -> TSPSolutionArrays:
        N = self.instance.num_nodes
        next = self._next.reshape(-1, N).detach().cpu().contiguous().numpy()
        prev = self._prev.reshape(-1, N).detach().cpu().contiguous().numpy()
        return TSPSolutionArrays(next=next, prev=prev)

    @classmethod
    def from_numpy(
        cls,
        instance: TSPInstance,
        solution: TSPSolutionArrays,
        *,
        batch_size: torch.Size,
        device: torch.device,
    ) -> TSPGraphSolution:
        num_nodes = instance.num_nodes
        next_tensor = (
            torch.from_numpy(solution.next)
            .to(device=device, dtype=torch.long)
            .reshape(*batch_size, num_nodes)
        )
        prev_tensor = (
            torch.from_numpy(solution.prev)
            .to(device=device, dtype=torch.long)
            .reshape(*batch_size, num_nodes)
        )
        return cls(_instance=instance, _next=next_tensor, _prev=prev_tensor, batch_size=batch_size)

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

    def where(self, mask: torch.Tensor, other: TSPGraphSolution) -> TSPGraphSolution:
        m = mask.unsqueeze(-1)
        return TSPGraphSolution(
            _instance=self.instance,
            _next=torch.where(m, self._next, other._next),
            _prev=torch.where(m, self._prev, other._prev),
            batch_size=self.batch_size,
        )

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

    def cumulative_tour_length(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative travel distances and tour total per node.

        The cycle is broken at node 0: its incoming edge is counted in total
        but zeroed in acc so that acc[0] = 0 (the fixed origin).

        Returns:
            acc:   (..., N) cumulative distance from node 0 along the tour.
            total: (..., N) full tour length broadcast to all nodes.
        """
        locations = self._instance.locations  # (..., N, 2)
        n = self._instance.num_nodes

        # Incoming edge length per node: ||x_v - x_prev[v]||. Unassigned nodes
        # (prev < 0) contribute 0.
        safe_prev = self._prev.clamp(min=0)
        prev_loc = locations.gather(-2, safe_prev.unsqueeze(-1).expand(*safe_prev.shape, 2))
        edge_len = (locations - prev_loc).norm(dim=-1) * (self._prev >= 0)  # (..., N)

        # Full tour length: every edge is the incoming edge of exactly one node.
        # Computed before the cycle-break so node 0's incoming edge is included.
        total = edge_len.sum(dim=-1, keepdim=True).expand_as(edge_len)  # (..., N)

        # Break the cycle at the origin (node 0): zero its incoming edge and make it
        # self-loop so pointer doubling terminates there.
        acc = edge_len.clone()
        ptr = safe_prev.clone()
        acc[..., 0] = 0.0
        ptr[..., 0] = 0

        for _ in range(math.ceil(math.log2(n + 1))):
            acc = acc + acc.gather(-1, ptr)
            ptr = ptr.gather(-1, ptr)

        return acc, total

    def tour_phase(self) -> torch.Tensor:
        """(..., N) float - in-tour position d_hat in [0, 2*pi] per node.

        The cumulative travel distance from a fixed origin (node 0) along the tour,
        rescaled by the full tour length (the in-route component of the hierarchical
        positional encoding for the single-route TSP case). Computed from the
        next/prev pointers via pointer doubling; the cycle is broken at the origin.
        """
        acc, total = self.cumulative_tour_length()
        return 2 * math.pi * acc / total.clamp_min(EPS)

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
        with_direction: bool = True,
        separator: str = " | ",
        scale: float = 1.0,
    ) -> None:
        from .render import render_solution

        render_solution(
            self,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            ax=ax,
            with_problem_size=with_problem_size,
            with_objective=with_objective,
            with_direction=with_direction,
            separator=separator,
            scale=scale,
        )

    @property
    def is_complete(self) -> torch.Tensor:
        return self.num_unassigned == 0

    def remove_nodes(self, nodes: torch.Tensor, *, stitch: bool = False) -> TSPGraphSolution:
        target_batch_shape = nodes.shape[:-1]

        shape_diff = len(target_batch_shape) - len(self.batch_size)
        expanded = (
            self.reshape(*self.batch_size, *[1] * shape_diff).expand(target_batch_shape).clone()
        )

        if stitch:
            for m in range(nodes.shape[-1]):
                expanded._remove_node_(nodes[..., m])
            return expanded

        predecessors = expanded._prev.gather(-1, nodes)
        successors = expanded._next.gather(-1, nodes)
        expanded._next.scatter_(-1, predecessors, self.UNASSIGNED)
        expanded._prev.scatter_(-1, nodes, self.UNASSIGNED)
        expanded._next.scatter_(-1, nodes, self.UNASSIGNED)
        expanded._prev.scatter_(-1, successors, self.UNASSIGNED)

        return expanded

    def _remove_node_(self, node: torch.Tensor) -> None:
        """Removes one node per batch element, stitching its neighbours together. In-place."""
        node_idx = node.unsqueeze(-1)
        u = self._prev.gather(-1, node_idx)
        v = self._next.gather(-1, node_idx)
        safe_u = u.clamp(min=0)
        safe_v = v.clamp(min=0)

        self._next.scatter_(-1, safe_u, v)
        self._prev.scatter_(-1, safe_v, u)

        self._prev.scatter_(-1, node_idx, node_idx.new_full(node_idx.shape, self.UNASSIGNED))
        self._next.scatter_(-1, node_idx, node_idx.new_full(node_idx.shape, self.UNASSIGNED))

    def insertion_positions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All existing directed edges in the solution.

        Returns:
            u: (..., N) source nodes. Invalid slots use 0.
            v: (..., N) target nodes, i.e. ``arange(N)``.
            valid: (..., N) bool - True where node ``v`` has an assigned predecessor.
        """
        n = self._instance.num_nodes
        v = torch.arange(n, device=self._next.device).expand(*self.batch_size, n)
        valid = self._prev != self.UNASSIGNED
        u = self._prev.clamp(min=0)
        return u, v, valid

    def insert_(
        self,
        c: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Inserts node c between predecessor u and successor v. In-place.

        Args:
            c: Node indices to insert, shape (...,).
            u: Predecessor indices, shape (...,).
            v: Successor indices, shape (...,).
            mask: Optional (...,) bool. If provided, only batches where the mask
                is True are updated; others keep their previous state.
        """
        c_idx = c.unsqueeze(-1)
        u_idx = u.unsqueeze(-1)
        v_idx = v.unsqueeze(-1)
        active = (
            mask.unsqueeze(-1) if mask is not None else torch.ones_like(c_idx, dtype=torch.bool)
        )

        self._prev.scatter_(-1, c_idx, torch.where(active, u_idx, self._prev.gather(-1, c_idx)))
        self._next.scatter_(-1, c_idx, torch.where(active, v_idx, self._next.gather(-1, c_idx)))
        self._next.scatter_(-1, u_idx, torch.where(active, c_idx, self._next.gather(-1, u_idx)))
        self._prev.scatter_(-1, v_idx, torch.where(active, c_idx, self._prev.gather(-1, v_idx)))

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
