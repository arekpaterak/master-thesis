from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...constants import EPS
from ..solution import BaseSolution
from .instance import CVRPInstance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class CVRPSolution(BaseSolution):
    """Canonical representation of single or batched CVRP solution."""

    UNASSIGNED = -1
    DEPOT = 0

    instance: CVRPInstance
    sequence: torch.Tensor
    """(..., L) where L is the max sequence length in the batch. Starts and ends with 0 (depot). Padded with 0s if necessary. -1 for deconstructed slots."""

    def __post_init__(self):
        if self.sequence is not None:
            inferred_batch_size = self.sequence.shape[:-1]
            if self.batch_size != inferred_batch_size:
                self.batch_size = inferred_batch_size

        assert self.batch_size == self.instance.batch_size

        # Validate that sequence indices are within bounds
        if self.sequence.numel() > 0:
            assert (self.sequence >= self.UNASSIGNED).all(), (
                "Node indices must be non-negative or be -1."
            )
            assert (self.sequence <= self.instance.num_customers).all(), "Node index out of bounds."

        assert (self.sequence[..., 0] == self.DEPOT).all() and (
            self.sequence[..., -1] == self.DEPOT
        ).all(), "Sequence must start and end at depot (0)."

    def objective(self) -> torch.Tensor:
        return self.total_distance()

    def total_distance(self) -> torch.Tensor:
        """
        Calculates total distance for single or batched instances.
        Computes the norm between consecutive nodes in the sequence.
        """
        # (..., L, 2) - Gather locations based on sequence indices
        # We need to expand sequence to match the (x, y) dimension
        idx = self.sequence.unsqueeze(-1).expand(*self.sequence.shape, 2)
        points = self.instance.locations.gather(dim=-2, index=idx)

        # Distance between consecutive points in the sequence
        return (points[..., 1:, :] - points[..., :-1, :]).norm(dim=-1).sum(dim=-1)

    @property
    def is_complete(self) -> torch.Tensor:
        sequence = self.sequence
        num_customers = self.instance.num_customers

        customer_mask = (sequence >= 1) & (sequence <= num_customers)
        one_hot = torch.nn.functional.one_hot(
            sequence.clamp(min=0, max=num_customers), num_classes=num_customers + 1
        )[..., 1:]
        customer_counts = (one_hot * customer_mask.unsqueeze(-1)).sum(dim=-2)
        each_customer_once = (customer_counts == 1).all(dim=-1)

        return ~self.is_deconstructed & each_customer_once

    @property
    def is_deconstructed(self) -> torch.Tensor:
        return (self.sequence == self.UNASSIGNED).any(dim=-1)

    @property
    def is_feasible(self) -> torch.Tensor:
        demand_per_route = self.demand_per_route()

        capacity = self.instance.capacity

        return self.is_complete & (demand_per_route <= capacity + EPS).all(dim=-1)

    @property
    def route_indexed_sequence(self):
        result = (self.sequence == self.DEPOT).long().cumsum(dim=-1) - 1

        # Ensure the final depot is part of the last route index
        result[..., -1] = result[..., -2]
        return result

    @property
    def num_routes(self) -> torch.Tensor:
        return self.route_indexed_sequence.max(dim=-1).values + 1

    def demand_per_route(self):
        route_idx = self.route_indexed_sequence
        demand = self.instance.demand.gather(-1, self.sequence)
        assert demand.shape == self.sequence.shape

        max_num_routes = self.num_routes.max().item()
        output_size = (*self.batch_size, max_num_routes)

        return torch.zeros(
            output_size, device=self.sequence.device, dtype=demand.dtype
        ).scatter_reduce(dim=-1, index=route_idx, src=demand, reduce="sum", include_self=False)

    def route_per_customer(self) -> torch.Tensor:
        """
        Returns a tensor of shape (..., num_customers) where each entry
        at index i is the route index for customer i+1.
        """
        num_customers = self.instance.num_customers
        route_indexed_sequence = self.route_indexed_sequence

        # Create a mask for valid customers (nodes 1 to N+1)
        customer_mask = (self.sequence >= 1) & (self.sequence <= num_customers)

        # Map customer 1..N+1 to indices 0..N.
        # Map everything else to a 'trash bin' at index N+1.
        trash_bin_idx = num_customers
        target_indices = torch.where(
            condition=customer_mask,
            input=self.sequence - 1,  # customer indexed
            other=torch.tensor(trash_bin_idx, device=self.sequence.device),
        )

        # -1 as a default to identify unvisited customers
        buffer = torch.full(
            (*self.batch_size, num_customers + 1),
            -1,
            dtype=route_indexed_sequence.dtype,
            device=self.sequence.device,
        )

        # Scatter the route IDs into the buffer
        buffer.scatter_(
            dim=-1,
            index=target_indices,
            src=route_indexed_sequence,
        )

        # Return only the customer part (exclude the trash bin)
        return buffer[..., :num_customers]

    def neighbours_per_customer(self) -> torch.Tensor:
        """
        Returns a tensor of shape (..., num_customers, 2) where:

        - [..., i, 0] is the predecessor of customer i+1

        - [..., i, 1] is the successor of customer i+1
        """
        num_customers = self.instance.num_customers
        prefix_shape = self.sequence.shape[:-1]

        predecessors = self.sequence.roll(shifts=1, dims=-1)
        successors = self.sequence.roll(shifts=-1, dims=-1)
        neighbour_pairs = torch.stack([predecessors, successors], dim=-1)

        customer_mask = (self.sequence >= 1) & (self.sequence <= num_customers)

        trash_bin_idx = num_customers
        target_indices = torch.where(
            condition=customer_mask,
            input=self.sequence - 1,
            other=torch.tensor(trash_bin_idx, device=self.sequence.device),
        )

        # Prepare the buffer (..., num_customers + 1, 2)
        buffer = torch.zeros(
            (*prefix_shape, num_customers + 1, 2),
            dtype=self.sequence.dtype,
            device=self.sequence.device,
        )

        # Scatter neighbour pairs into the buffer
        # Expand target_indices to (..., L, 2) to match neighbour_pairs
        expanded_target = target_indices.unsqueeze(-1).expand_as(neighbour_pairs)
        buffer.scatter_(dim=-2, index=expanded_target, src=neighbour_pairs)

        # 5. Return only the customer indices (excluding trash bin)
        return buffer[..., :num_customers, :]

    def remove_customers(self, to_remove: torch.Tensor) -> CVRPSolution:
        """
        Deconstruct the solution.

        Args:
            to_remove: (batch_size, deconstruction_size) or (batch_size, rollout_size, deconstruction_size)

        Returns:
            A new CVRPSolution with the specified customers removed (set to -1 in the sequence).
        """
        # 1. Determine the target batch shape by broadcasting the leading dimensions
        # This handles cases where to_remove has extra dims like (batch, rollout)
        target_batch_shape = torch.broadcast_shapes(self.batch_size, to_remove.shape[:-1])

        # 2. Expand sequence and instance to match the new batch shape
        sequence = self.sequence.expand(*target_batch_shape, -1)
        instance = self.instance.expand(*target_batch_shape)

        # 3. Create a mask identifying where nodes in the sequence match nodes in to_remove
        # We compare (..., L, 1) with (..., 1, K) -> (..., L, K)
        # .any(-1) reduces it to (..., L)
        is_removed = (sequence.unsqueeze(-1) == to_remove.unsqueeze(-2)).any(dim=-1)

        # 4. Safety: Only allow removal of customers (node indices >= 1)
        # This prevents accidental removal of depots (0) or padding (-1)
        mask = is_removed & (sequence >= 1)

        # 5. Create the new sequence with -1 at masked positions
        new_sequence = sequence.clone().masked_fill_(mask, -1)

        return CVRPSolution(instance=instance, sequence=new_sequence, batch_size=target_batch_shape)

    def routes(self) -> list[torch.Tensor]:
        """Splits the single-instance sequence into a list of routes (excluding depot)."""
        if self.batch_size.numel() > 1:
            raise ValueError("routes() only works on a single instance.")

        sequence = self.sequence.cpu()
        depot_indices = (sequence == self.DEPOT).nonzero(as_tuple=True)[0]
        routes = []
        for start, end in zip(depot_indices[:-1], depot_indices[1:]):
            route = sequence[start + 1 : end]  # skip depots
            if route.numel() > 0:
                routes.append(route)
        return routes

    def visualize(
        self,
        *,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
        with_problem_size: bool = True,
        with_objective: bool = True,
        show_depot_edges: bool = False,
        depot_in_legend: bool = True,
        with_route_labels: bool = True,
        legend_loc: str | None = "upper right",
    ) -> None:
        from .render import render_solution

        render_solution(
            self,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            ax=ax,
            with_problem_size=with_problem_size,
            with_objective=with_objective,
            show_depot_edges=show_depot_edges,
            depot_in_legend=depot_in_legend,
            with_route_labels=with_route_labels,
            legend_loc=legend_loc,
        )
