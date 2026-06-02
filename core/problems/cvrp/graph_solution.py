from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..solution import BaseSolution
from .instance import CVRPInstance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class CVRPGraphSolution(BaseSolution):
    """Edge-based representation of a CVRP solution using next/prev pointer tensors."""

    UNASSIGNED = -1
    DEPOT = 0

    _instance: CVRPInstance
    _next: torch.Tensor
    """(..., N+1) successor per node. 0 = depot, 1..N = customer, -1 = unassigned."""
    _prev: torch.Tensor
    """(..., N+1) predecessor per node. 0 = depot, 1..N = customer, -1 = unassigned."""
    _route_id: torch.Tensor
    """(..., N+1) route ID per node.

    Always -1 for the depot (index 0). Always >= 1 for customers: equals the
    node index of the route's head customer (the one with prev <= 0). Isolated
    customers after destroy are their own route head, so route_id equals their
    own index. During incremental repair the min-convention is used: when two
    routes merge, the smaller head index wins.
    """

    @classmethod
    def create_from_sequence(
        cls, instance: CVRPInstance, sequence: torch.Tensor
    ) -> CVRPGraphSolution:
        """Builds a graph solution from a CVRPSolution-style flat sequence.

        Args:
            instance: The CVRP problem instance.
            sequence: (..., L) long tensor. 0=depot, 1..N=customer.
        """
        assert not (sequence == cls.UNASSIGNED).any(), "Expected complete sequence without -1"

        batch_size = instance.batch_size
        device = sequence.device

        _next = torch.full(
            (*batch_size, instance.num_nodes), cls.UNASSIGNED, dtype=torch.long, device=device
        )
        _prev = torch.full(
            (*batch_size, instance.num_nodes), cls.UNASSIGNED, dtype=torch.long, device=device
        )

        _next.scatter_(-1, sequence, sequence.roll(-1, dims=-1))
        _prev.scatter_(-1, sequence, sequence.roll(1, dims=-1))
        _next[..., 0] = cls.UNASSIGNED
        _prev[..., 0] = cls.UNASSIGNED

        _route_id = cls._compute_route_ids(_prev, instance.num_customers)
        return cls(
            _instance=instance,
            _next=_next,
            _prev=_prev,
            _route_id=_route_id,
            batch_size=batch_size,
        )

    @classmethod
    def one_customer_per_route(cls, instance: CVRPInstance) -> CVRPGraphSolution:
        """Builds a solution with each customer in its own single-node route.

        Args:
            instance: The CVRP problem instance.
        """
        N = instance.num_customers
        batch_size = instance.batch_size
        device = instance.locations.device

        _next = torch.full((*batch_size, N + 1), cls.DEPOT, dtype=torch.long, device=device)
        _prev = torch.full((*batch_size, N + 1), cls.DEPOT, dtype=torch.long, device=device)
        _next[..., 0] = cls.UNASSIGNED
        _prev[..., 0] = cls.UNASSIGNED

        # Each customer connects directly to the depot, so it is its own route head.
        _route_id = (
            torch.arange(N + 1, dtype=torch.long, device=device).expand(*batch_size, N + 1).clone()
        )
        _route_id[..., 0] = cls.UNASSIGNED

        return cls(
            _instance=instance,
            _next=_next,
            _prev=_prev,
            _route_id=_route_id,
            batch_size=batch_size,
        )

    @staticmethod
    def _compute_route_ids(prev: torch.Tensor, num_customers: int) -> torch.Tensor:
        """Recomputes route IDs from scratch.

        Route members (prev > 0) follow their predecessor until reaching a route head.
        Route heads (prev == 0, depot-connected) and isolated/fragment heads
        (prev == -1) both self-loop, converging to their own node index. All
        customers therefore get a valid positive route_id >= 1.

        O(N log N) computation, fully vectorised. Intended for use after destroy, not
        during incremental repair.

        Args:
            prev: Predecessor pointers of shape (..., N+1).
            num_customers: N, excluding the depot.

        Returns:
            Route ID tensor of shape (..., N+1). Only the depot (index 0) is -1.
        """
        V = num_customers + 1
        device = prev.device
        batch_shape = prev.shape[:-1]
        arange = torch.arange(V, device=device).expand(*batch_shape, V)

        # prev > 0: follow predecessor. prev <= 0 (depot or -1): self-loop.
        parent = torch.where(prev > 0, prev, arange)

        for _ in range(math.ceil(math.log2(V + 1))):
            parent = parent.gather(-1, parent)

        route_id = parent.clone()
        route_id[..., 0] = CVRPGraphSolution.UNASSIGNED
        return route_id

    def __post_init__(self) -> None:
        assert self._next.shape == self._prev.shape == self._route_id.shape

        N = self.instance.num_customers

        assert ((self._next >= -1) & (self._next <= N)).all(), "Invalid node index in _next"
        assert ((self._prev >= -1) & (self._prev <= N)).all(), "Invalid node index in _prev"

        # Customer rules
        assert (self._route_id[..., 1:] >= 0).all(), "Customer Route IDs must be >= 0"

        # Depot rules
        assert (self._route_id[..., 0] == self.UNASSIGNED).all(), "Depot Route ID must be -1"
        assert (self._next[..., 0] == self.UNASSIGNED).all(), "Depot _next should be unassigned"
        assert (self._prev[..., 0] == self.UNASSIGNED).all(), "Depot _prev should be unassigned"

    @property
    def instance(self) -> CVRPInstance:
        return self._instance

    @property
    def next_nodes(self) -> torch.Tensor:
        return self._next

    @property
    def prev_nodes(self) -> torch.Tensor:
        return self._prev

    @property
    def route_ids(self) -> torch.Tensor:
        return self._route_id

    @property
    def is_complete(self) -> torch.Tensor:
        return self.num_unassigned == 0

    @property
    def is_feasible(self) -> torch.Tensor:
        """(...) bool - complete AND every route is within capacity."""
        route_demands = self.demand_per_route()
        return self.is_complete & (route_demands <= self._instance.capacity).all(dim=-1)

    @property
    def num_unassigned(self) -> torch.Tensor:
        """(...) int - number of customers missing a successor OR predecessor."""
        missing_next = self._next[..., 1:] == self.UNASSIGNED
        missing_prev = self._prev[..., 1:] == self.UNASSIGNED
        return (missing_next | missing_prev).sum(dim=-1)

    @property
    def num_routes(self) -> torch.Tensor:
        """(...) int - number of route chains in the current solution.

        Counts customers with prev <= 0: depot-connected heads (prev == 0) and
        fragment/isolated heads (prev == -1, after destroy or during repair).
        """
        return (self._prev[..., 1:] <= self.DEPOT).sum(dim=-1)

    @property
    def max_routes(self) -> int:
        """Maximum number of routes across the batch."""
        return int(self.num_routes.max().item())

    def node_degree(self) -> torch.Tensor:
        """(..., N) int - degree per customer (0, 1, or 2)."""
        has_next = (self._next[..., 1:] != self.UNASSIGNED).long()
        has_prev = (self._prev[..., 1:] != self.UNASSIGNED).long()
        return has_next + has_prev

    def neighbours_per_customer(self) -> torch.Tensor:
        """(..., N, 2) int - [prev, next] per customer. -1 if unassigned."""
        return torch.stack([self._prev[..., 1:], self._next[..., 1:]], dim=-1)

    def route_per_customer(self) -> torch.Tensor:
        """(..., N) int - route ID for each customer. Always >= 1."""
        return self._route_id[..., 1:]

    def demand_per_route(self) -> torch.Tensor:
        """(..., max_route_id+1) float - total demand accumulated per route.

        Route IDs are head-node indices (sparse in 1..N), so the output tensor
        is sized to max(route_id)+1. All customers always have a valid route_id >= 1.
        """
        route_ids = self._route_id[..., 1:]  # (..., N), all >= 1
        demands = self._instance.demand[..., 1:]  # (..., N)
        max_id = int(route_ids.max().item()) + 1
        out = torch.zeros(*route_ids.shape[:-1], max_id, dtype=demands.dtype, device=demands.device)
        out.scatter_add_(-1, route_ids, demands)
        return out

    def objective(self) -> torch.Tensor:
        """(...) float - total tour distance across all routes."""
        assert self.is_complete.all(), (
            "objective() requires all customers to be assigned; "
            f"found {self.num_unassigned.max().item()} unassigned in at least one batch element"
        )
        locations = self._instance.locations  # (..., N+1, 2)
        from_locs = locations[..., 1:, :]  # (..., N, 2)

        # Customer -> next edges (next may be depot=0 or another customer)
        has_next = self._next[..., 1:] >= 0  # (..., N)
        safe_next = self._next[..., 1:].clamp(min=0)  # (..., N)
        to_locs = locations.gather(-2, safe_next.unsqueeze(-1).expand(*safe_next.shape, 2))
        edge_dists = (from_locs - to_locs).pow(2).sum(-1).sqrt() * has_next

        # Depot -> first customer edges (route heads have prev == depot)
        is_head = self._prev[..., 1:] == self.DEPOT  # (..., N)
        depot_loc = locations[..., :1, :]  # (..., 1, 2)
        depot_dists = (from_locs - depot_loc).pow(2).sum(-1).sqrt() * is_head

        return (edge_dists + depot_dists).sum(-1)

    def remove_customers(self, customers: torch.Tensor) -> CVRPGraphSolution:
        """Removes customers out of their routes and recomputes route IDs.

        Args:
            customers: Indices of customers to remove, shape (..., M). Values in 1..N.

        Returns:
            New CVRPGraphSolution with the customers detached and route_id updated.
        """
        assert (customers >= 1).all(), "Expected only customers (>= 1)"

        target_batch_shape = customers.shape[:-1]

        shape_diff = len(target_batch_shape) - len(self.batch_size)
        expanded = (
            self.reshape(*self.batch_size, *[1] * shape_diff).expand(target_batch_shape).clone()
        )

        predecessors = expanded._prev.gather(-1, customers)
        successors = expanded._next.gather(-1, customers)

        safe_p = predecessors.clamp(min=0)
        safe_n = successors.clamp(min=0)

        expanded._next.scatter_(
            -1,
            safe_p,
            torch.where(predecessors > 0, self.UNASSIGNED, expanded._next.gather(-1, safe_p)),
        )
        expanded._prev.scatter_(-1, customers, self.UNASSIGNED)
        expanded._next.scatter_(-1, customers, self.UNASSIGNED)
        expanded._prev.scatter_(
            -1,
            safe_n,
            torch.where(successors > 0, self.UNASSIGNED, expanded._prev.gather(-1, safe_n)),
        )

        route_id = self._compute_route_ids(expanded.prev_nodes, expanded.instance.num_customers)
        return CVRPGraphSolution(
            _instance=expanded._instance,
            _next=expanded.next_nodes,
            _prev=expanded.prev_nodes,
            _route_id=route_id,
            batch_size=expanded.batch_size,
        )

    def remove_nodes(self, nodes: torch.Tensor) -> CVRPGraphSolution:
        """
        Alias for remove_customers, since in CVRP we only remove customers (not depots).
        """
        return self.remove_customers(nodes)

    def add_edge_(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Adds directed edge u -> v and merges route IDs. In-place.

        Args:
            u: Tail node indices, shape (...,). 0 = depot, else must satisfy next[u] == -1.
            v: Head node indices, shape (...,). 0 = depot, else must satisfy prev[v] <= 0.
            mask: Optional (...,) bool. If provided, only batches where the mask is True
                are updated; others keep their previous state. Used by repair to gate
                writes for batches that already have no remaining work.
        """
        u_idx = u.unsqueeze(-1)
        v_idx = v.unsqueeze(-1)

        if mask is None:
            self._next.scatter_(-1, u_idx, v_idx)
            self._prev.scatter_(-1, v_idx, u_idx)
        else:
            m = mask.unsqueeze(-1)
            old_next_u = self._next.gather(-1, u_idx)
            old_prev_v = self._prev.gather(-1, v_idx)
            self._next.scatter_(-1, u_idx, torch.where(m, v_idx, old_next_u))
            self._prev.scatter_(-1, v_idx, torch.where(m, u_idx, old_prev_v))
        self._next[..., 0] = self.UNASSIGNED
        self._prev[..., 0] = self.UNASSIGNED

        # Route ID merge only applies to customer-to-customer edges; depot has no route_id.
        both_customers = (u_idx > 0) & (v_idx > 0)  # (..., 1)
        if mask is not None:
            both_customers = both_customers & mask.unsqueeze(-1)
        if both_customers.any():
            u_route_id = self._route_id.gather(-1, u_idx)  # (..., 1)
            v_route_id = self._route_id.gather(-1, v_idx)  # (..., 1)
            merged = torch.minimum(u_route_id, v_route_id)
            absorbed = torch.maximum(u_route_id, v_route_id)
            new_route_id = torch.where(self._route_id == absorbed, merged, self._route_id)
            self._route_id = torch.where(both_customers, new_route_id, self._route_id)

    def add_edge(self, u: torch.Tensor, v: torch.Tensor) -> CVRPGraphSolution:
        solution = self.clone()
        solution.add_edge_(u, v)
        return solution

    def add_edges(self, edges: torch.Tensor) -> CVRPGraphSolution:
        """Adds multiple directed edges sequentially.

        Args:
            edges: Shape (..., E, 2).

        Returns:
            New CVRPGraphSolution with all edges applied in order.
        """
        solution = self.clone()
        for e in range(edges.shape[-2]):
            solution.add_edge_(edges[..., e, 0], edges[..., e, 1])
        return solution

    def routes(self) -> list[torch.Tensor]:
        """Splits the single-instance solution into a list of route tensors.

        Each tensor contains the customer indices (1..N) of one route, in traversal order.
        Routes without a depot connection on either end are included as partial routes.
        """
        if self.batch_size.numel() > 1:
            raise ValueError(
                f"routes() only works on a single instance. "
                f"Current batch_size is {self.batch_size}. Try solution[0].routes()."
            )
        solution = self.squeeze(0) if self.batch_size else self
        next_nodes = solution._next.tolist()
        prev_nodes = solution._prev.tolist()

        N = solution.instance.num_customers
        routes = []
        for head in range(1, N + 1):
            if prev_nodes[head] <= 0:
                route, node = [], head
                while node >= 1:
                    route.append(node)
                    node = next_nodes[node]
                routes.append(torch.tensor(route, dtype=torch.long))
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
        with_route_labels: bool = False,
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
