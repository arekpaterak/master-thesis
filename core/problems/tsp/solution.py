from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.problems.solution import BaseSolution
from core.problems.tsp import TSPInstance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class TSPSolution(BaseSolution):
    """
    Canonical representation of single or batched TSP solution.
    """

    _instance: TSPInstance
    _tour: torch.Tensor
    """(..., n), -1 for unvisited slots."""

    UNASSIGNED = -1

    @property
    def instance(self) -> TSPInstance:
        return self._instance

    @property
    def tour(self) -> torch.Tensor:
        return self._tour

    @classmethod
    def create(cls, instance: TSPInstance, tour: torch.Tensor) -> TSPSolution:
        return cls(_instance=instance, _tour=tour, batch_size=instance.batch_size)

    def __post_init__(self):
        mask = self.tour != self.UNASSIGNED
        if mask.any():
            assert self.tour[mask].max() < self.instance.num_locations, (
                "Indices in tour should be less than the number of locations in the instance."
            )
            assert self.tour[mask].min() >= 0, "Indices in tour should be non-negative."

        unassigned_count = self.unassigned_mask.sum(dim=-1)
        assert (unassigned_count == unassigned_count.flatten()[0]).all(), (
            "Expected all instances to have the same number of unassigned nodes."
        )

    @property
    def num_visited(self) -> torch.Tensor:
        return (self.tour != self.UNASSIGNED).sum(dim=-1)

    @property
    def unassigned_mask(self) -> torch.Tensor:
        return self.tour == self.UNASSIGNED

    def unassigned_nodes(self) -> torch.Tensor:
        b_size = self.batch_size
        n = self.instance.num_locations
        all_nodes = torch.arange(n, device=self.tour.device).expand(*b_size, n)

        mask = self.tour_index() == self.UNASSIGNED
        unassigned_flat = all_nodes[mask]

        return unassigned_flat.view(*b_size, -1)

    @property
    def num_unassigned(self) -> torch.Tensor:
        return (self.tour == self.UNASSIGNED).sum(dim=-1)

    @property
    def is_complete(self) -> torch.Tensor:
        return self.num_visited == self.instance.num_locations

    @property
    def is_feasible(self) -> torch.Tensor:
        return self.is_complete  # All permutations are feasible

    def objective(self) -> torch.Tensor:
        if (self.is_complete == False).any():
            raise ValueError("Cannot calculate objective for incomplete TSP solutions.")
        return self._tour_length()

    def neighbours_index(self) -> torch.Tensor:
        b_size = self.batch_size
        n = self.instance.num_nodes
        tour = self.tour

        valid_mask = tour != self.UNASSIGNED
        trash_bin_idx = n

        neighbours_buffer = torch.full(
            (*b_size, n + 1, 2), -1, dtype=torch.long, device=tour.device
        )
        target_indices = torch.where(
            valid_mask, tour, torch.tensor(trash_bin_idx, device=tour.device)
        )

        # Valid inputs for successors and predecessors when the node itself is assigned
        # But if a node is unassigned, it should be -1.
        predecessors = torch.where(
            valid_mask, tour.roll(1, dims=-1), torch.tensor(-1, device=tour.device)
        )
        successors = torch.where(
            valid_mask, tour.roll(-1, dims=-1), torch.tensor(-1, device=tour.device)
        )

        neighbours_buffer[..., 0].scatter_(dim=-1, index=target_indices, src=predecessors)
        neighbours_buffer[..., 1].scatter_(dim=-1, index=target_indices, src=successors)

        return neighbours_buffer[..., :n, :]

    def tour_index(self) -> torch.Tensor:
        b_size = self.batch_size
        n = self.instance.num_nodes
        tour = self.tour

        valid_mask = tour != self.UNASSIGNED
        trash_bin_idx = n
        target_indices = torch.where(
            valid_mask, tour, torch.tensor(trash_bin_idx, device=tour.device)
        )

        # For TSP there's only 1 tour! Nodes in tour have index 0, unassigned nodes have -1.
        tour_index_buffer = torch.full((*b_size, n + 1), -1, dtype=torch.long, device=tour.device)
        tour_id = torch.zeros_like(tour)
        tour_index_buffer.scatter_(dim=-1, index=target_indices, src=tour_id)

        return tour_index_buffer[..., :n]

    @property
    def next_nodes(self) -> torch.Tensor:
        return self.neighbours_index()[..., 1]

    @property
    def prev_nodes(self) -> torch.Tensor:
        return self.neighbours_index()[..., 0]

    def node_degree(self) -> torch.Tensor:
        neighbours = self.neighbours_index()
        return (neighbours != self.UNASSIGNED).sum(dim=-1)

    def tour_phase(self) -> torch.Tensor:
        """Not supported on the sequence representation; use TSPGraphSolution."""
        raise NotImplementedError(
            "tour_phase is only implemented for TSPGraphSolution, not the sequence representation."
        )

    def remove_nodes(self, to_remove: torch.Tensor) -> TSPSolution:
        """
        Deconstruct the TSP solution.

        Args:
            to_remove: Tensor containing node indices to remove with shape [..., deconstruction_size].

        Returns:
            A new solution with the specified nodes replaced by -1.
        """
        # 1. Determine the target batch shape by broadcasting leading dimensions
        target_batch_shape = torch.broadcast_shapes(self.batch_size, to_remove.shape[:-1])

        # 2. Expand tour and instance to match the new batch shape
        tour = self.tour.expand(*target_batch_shape, -1)
        instance = self.instance.expand(*target_batch_shape)

        # 3. Create a mask identifying where nodes in the tour match nodes in to_remove
        # (..., num_nodes, 1) == (..., 1, deconstruction_size) -> (..., num_nodes)
        is_removed = (tour.unsqueeze(-1) == to_remove.unsqueeze(-2)).any(dim=-1)

        # 4. Create the new tour with -1 at masked positions
        new_tour = tour.clone().masked_fill_(is_removed, self.UNASSIGNED)

        return TSPSolution.create(instance=instance, tour=new_tour)

    def _tour_length(self) -> torch.Tensor:
        """
        Handles both complete and partial solutions.
        """
        locations = self.instance.locations

        safe_tour = self.tour.clamp(min=0)

        # (..., N, 2) coordinates in tour order
        ordered_locs = locations.gather(
            dim=-2, index=safe_tour.unsqueeze(-1).expand(*safe_tour.shape, 2)
        )

        # 1. Calculate distances between consecutive nodes in the sequence
        p1 = ordered_locs[..., :-1, :]
        p2 = ordered_locs[..., 1:, :]
        dists = (p1 - p2).norm(dim=-1)

        # Mask out transitions involving PADDING
        mask = (self.tour[..., :-1] != self.UNASSIGNED) & (self.tour[..., 1:] != self.UNASSIGNED)
        total_dist = (dists * mask.float()).sum(dim=-1)

        # 2. Add the return trip to the start only for complete tours (the Hamiltonian Cycle)
        complete_mask = self.is_complete
        if complete_mask.any():
            start_loc = ordered_locs[..., 0, :]
            end_loc = ordered_locs[..., -1, :]
            return_dist = (end_loc - start_loc).norm(dim=-1)
            total_dist = torch.where(complete_mask, total_dist + return_dist, total_dist)

        return total_dist

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
