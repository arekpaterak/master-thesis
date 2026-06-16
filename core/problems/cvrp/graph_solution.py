from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ...constants import EPS
from ..solution import BaseSolution
from .instance import CVRPInstance

if TYPE_CHECKING:
    import numpy as np
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

    def to_numpy_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract contiguous numpy arrays for a numba improvement kernel.

        Batch dims are collapsed to a single leading dim G; floats are float64.
        Pair with :meth:`from_arrays` to rebuild a solution from the kernel output.

        Returns:
            ``(next, prev, locations, demand, capacity)`` with shapes
            ``(G, N+1)`` int64, ``(G, N+1)`` int64, ``(G, N+1, 2)`` float64,
            ``(G, N+1)`` float64 and ``(G,)`` float64.
        """
        N = self.instance.num_customers
        instance = self.instance
        next = self._next.reshape(-1, N + 1).detach().cpu().contiguous().numpy()
        prev = self._prev.reshape(-1, N + 1).detach().cpu().contiguous().numpy()
        locations = (
            instance.locations.reshape(-1, N + 1, 2).detach().cpu().double().contiguous().numpy()
        )
        demand = instance.demand.reshape(-1, N + 1).detach().cpu().double().contiguous().numpy()
        capacity = instance.capacity.reshape(-1).detach().cpu().double().contiguous().numpy()
        return next, prev, locations, demand, capacity

    @classmethod
    def from_numpy_arrays(
        cls,
        *,
        instance: CVRPInstance,
        next: np.ndarray,
        prev: np.ndarray,
        batch_size: torch.Size,
        device: torch.device,
    ) -> CVRPGraphSolution:
        """Rebuild a solution from kernel-produced ``next``/``prev`` numpy arrays.

        ``route_id`` is recomputed from ``prev`` so the result satisfies the class
        invariants without the kernel having to maintain it.
        """
        num_customers = instance.num_customers
        next_tensor = (
            torch.from_numpy(next)
            .to(device=device, dtype=torch.long)
            .reshape(*batch_size, num_customers + 1)
        )
        prev_tensor = (
            torch.from_numpy(prev)
            .to(device=device, dtype=torch.long)
            .reshape(*batch_size, num_customers + 1)
        )
        route_id = cls._compute_route_ids(prev_tensor, num_customers)
        return cls(
            _instance=instance,
            _next=next_tensor,
            _prev=prev_tensor,
            _route_id=route_id,
            batch_size=batch_size,
        )

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
        return self.is_complete & (route_demands <= self._instance.capacity + EPS).all(dim=-1)

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

    def insertion_positions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All directed edges in the solution, padded to 2N+1 slots.

        Slot layout:
            0:        synthetic new-route edge: (depot, depot); always valid.
            1..N:     incoming edges of customers: (prev[v], v) for v=1..N; valid when prev[v] != -1.
            N+1..2N:  outgoing-to-depot edges: (u, depot) for u=1..N; valid when next[u] == depot.

        Returns:
            u: (..., 2N+1) source nodes. Always >= 0; invalid slots use 0 (depot).
            v: (..., 2N+1) target nodes. Always >= 0.
            valid: (..., 2N+1) bool - True for slots that represent real or synthetic edges.
        """
        N = self.instance.num_customers
        batch_shape = self.batch_size
        device = self._next.device

        customers = torch.arange(1, N + 1, device=device).expand(*batch_shape, N)
        depot_single = torch.zeros(*batch_shape, 1, dtype=customers.dtype, device=device)

        u_in = self._prev.gather(-1, customers)  # (..., N); may be -1
        valid_in = u_in != self.UNASSIGNED
        out_valid = self._next.gather(-1, customers) == self.DEPOT
        new_valid = torch.ones(*batch_shape, 1, dtype=torch.bool, device=device)

        u = torch.cat([depot_single, u_in.clamp(min=0), customers], dim=-1)
        v = torch.cat([depot_single, customers, torch.zeros_like(customers)], dim=-1)
        valid = torch.cat([new_valid, valid_in, out_valid], dim=-1)
        return u, v, valid

    def route_per_customer(self) -> torch.Tensor:
        """(..., N) int - route ID for each customer. Always >= 1."""
        return self._route_id[..., 1:]

    def cumulative_route_length(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative travel distances and route totals per customer.

        Chains run from the opening depot edge to each customer; the depot
        (prev < 0) self-loops with zero edge length and terminates each chain.
        Route total = acc of the last customer in the route + its closing edge
        to the depot.

        Returns:
            acc:   (..., N) cumulative distance from the route's depot edge.
            total: (..., N) total route length for each customer's route.
        """
        locations = self._instance.locations  # (..., N+1, 2)
        N = self._instance.num_customers

        # Incoming edge length per node: ||x_v - x_prev[v]||. Depot and unassigned
        # nodes (prev < 0) contribute 0.
        safe_prev = self._prev.clamp(min=0)
        prev_loc = locations.gather(-2, safe_prev.unsqueeze(-1).expand(*safe_prev.shape, 2))
        edge_len = (locations - prev_loc).norm(dim=-1) * (self._prev >= 0)  # (..., N+1)

        # Cumulative distance from the route's depot edge to each node, via pointer
        # doubling along prev. The depot self-loops (safe_prev[..., 0] = 0) with
        # edge_len 0, so chains terminate there.
        acc_full = edge_len.clone()
        ptr = safe_prev.clone()
        for _ in range(math.ceil(math.log2(N + 2))):
            acc_full = acc_full + acc_full.gather(-1, ptr)
            ptr = ptr.gather(-1, ptr)

        acc = acc_full[..., 1:]  # (..., N)

        # Route total: acc of last customer + its closing edge to the depot,
        # bucketed by route_id then looked up per customer.
        cust_route = self._route_id[..., 1:]  # (..., N), all >= 1
        depot_loc = locations[..., :1, :]
        closing = (locations[..., 1:, :] - depot_loc).norm(dim=-1)  # (..., N)
        is_last = self._next[..., 1:] == self.DEPOT  # (..., N) bool

        route_total = torch.zeros(*cust_route.shape[:-1], N + 1, dtype=acc.dtype, device=acc.device)
        route_total.scatter_add_(-1, cust_route, (acc + closing) * is_last)
        total = route_total.gather(-1, cust_route)  # (..., N)

        return acc, total

    def route_phase(self) -> torch.Tensor:
        """(..., N) float - in-route position d_hat in [0, 2*pi] per customer.

        The cumulative travel distance from the route's opening depot edge to the
        customer, rescaled so a full closed route maps back to 2*pi (the in-route
        component of the hierarchical positional encoding). Computed from the
        next/prev pointers via the same pointer-doubling idiom as
        _compute_route_ids, so it needs no explicit route ordering.

        An isolated single-customer route has phase pi (the depot->customer edge is
        half of the depot->customer->depot loop). Degenerate zero-length routes map
        to 0.
        """
        acc, total = self.cumulative_route_length()
        return 2 * math.pi * acc / total.clamp_min(EPS)

    def demand_per_route(self) -> torch.Tensor:
        """(..., N+1) float - total demand accumulated per route.

        Route IDs are head-node indices (sparse in 1..N). Sized statically to
        N+1 to avoid a host sync on max(route_id); unused indices stay at 0.
        All customers always have a valid route_id >= 1.
        """
        route_ids = self._route_id[..., 1:]  # (..., N), all >= 1
        demands = self._instance.demand[..., 1:]  # (..., N)
        N = self._instance.num_customers
        out = torch.zeros(*route_ids.shape[:-1], N + 1, dtype=demands.dtype, device=demands.device)
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

    def remove_customers(
        self, customers: torch.Tensor, *, stitch: bool = False
    ) -> CVRPGraphSolution:
        """Removes customers out of their routes and recomputes route IDs.

        Args:
            customers: Indices of customers to remove, shape (..., M). Values in 1..N.
            stitch: If True, stitch each affected route's survivors together
                so the route remains a continuous depot->...->depot (no splitting).
                If False (default), the pointers surrounding each
                removed customer are left UNASSIGNED, producing fragments that a
                reconnect-style repair can join.

        Returns:
            New CVRPGraphSolution with the customers detached and route_id updated.
        """
        assert (customers >= 1).all(), "Expected only customers (>= 1)"

        target_batch_shape = customers.shape[:-1]
        shape_diff = len(target_batch_shape) - len(self.batch_size)
        expanded = (
            self.reshape(*self.batch_size, *[1] * shape_diff).expand(target_batch_shape).clone()
        )

        if stitch:
            return self._remove_customers_stitched(expanded, customers)
        return self._remove_customers_fragmented(expanded, customers)

    @staticmethod
    def _remove_customers_fragmented(
        expanded: CVRPGraphSolution, customers: torch.Tensor
    ) -> CVRPGraphSolution:
        predecessors = expanded._prev.gather(-1, customers)
        successors = expanded._next.gather(-1, customers)
        safe_p = predecessors.clamp(min=0)
        safe_n = successors.clamp(min=0)

        unassigned = torch.full_like(customers, CVRPGraphSolution.UNASSIGNED)
        expanded._next.scatter_(
            -1,
            safe_p,
            torch.where(predecessors > 0, unassigned, expanded._next.gather(-1, safe_p)),
        )
        expanded._prev.scatter_(
            -1,
            safe_n,
            torch.where(successors > 0, unassigned, expanded._prev.gather(-1, safe_n)),
        )
        expanded._prev.scatter_(-1, customers, CVRPGraphSolution.UNASSIGNED)
        expanded._next.scatter_(-1, customers, CVRPGraphSolution.UNASSIGNED)

        route_id = CVRPGraphSolution._compute_route_ids(
            expanded._prev, expanded.instance.num_customers
        )
        return CVRPGraphSolution(
            _instance=expanded._instance,
            _next=expanded._next,
            _prev=expanded._prev,
            _route_id=route_id,
            batch_size=expanded.batch_size,
        )

    def _remove_customer_(self, c: torch.Tensor, *, stitch: bool) -> None:
        """Removes one customer per batch element, optionally stitching neighbours. In-place.

        Args:
            c: Customer indices, shape (...,). Values in 1..N.
            stitch: Must be True; raises NotImplementedError otherwise.
        """
        if not stitch:
            raise NotImplementedError("_remove_customer_ only supports stitch=True")

        c_idx = c.unsqueeze(-1)
        u = self._prev.gather(-1, c_idx)
        v = self._next.gather(-1, c_idx)
        safe_u = u.clamp(min=0)
        safe_v = v.clamp(min=0)

        # Stitch: next[u] = v when u is a customer; prev[v] = u when v is a customer
        self._next.scatter_(-1, safe_u, torch.where(u > 0, v, self._next.gather(-1, safe_u)))
        self._prev.scatter_(-1, safe_v, torch.where(v > 0, u, self._prev.gather(-1, safe_v)))

        # Clear removed customer's pointers
        self._prev.scatter_(-1, c_idx, c_idx.new_full(c_idx.shape, self.UNASSIGNED))
        self._next.scatter_(-1, c_idx, c_idx.new_full(c_idx.shape, self.UNASSIGNED))

        # Restore depot invariant
        self._next[..., 0] = self.UNASSIGNED
        self._prev[..., 0] = self.UNASSIGNED

        # Route-id: only for the removed node; survivors keep their route_id.
        self._route_id.scatter_(-1, c_idx, c_idx)

    @staticmethod
    def _remove_customers_stitched(
        expanded: CVRPGraphSolution, customers: torch.Tensor
    ) -> CVRPGraphSolution:
        for m in range(customers.shape[-1]):
            expanded._remove_customer_(customers[..., m], stitch=True)
        # TODO: Think if we can not recompute route ids. Now we have to because we might have removed route heads.
        route_id = CVRPGraphSolution._compute_route_ids(
            expanded._prev, expanded.instance.num_customers
        )
        return CVRPGraphSolution(
            _instance=expanded._instance,
            _next=expanded._next,
            _prev=expanded._prev,
            _route_id=route_id,
            batch_size=expanded.batch_size,
        )

    def remove_nodes(self, nodes: torch.Tensor, *, stitch: bool = False) -> CVRPGraphSolution:
        """
        Alias for remove_customers, since in CVRP we only remove customers (not depots).
        """
        return self.remove_customers(nodes, stitch=stitch)

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
                are updated; others keep their previous state.
        """
        u_idx = u.unsqueeze(-1)  # (..., 1)
        v_idx = v.unsqueeze(-1)  # (..., 1)
        active = mask.unsqueeze(-1) if mask is not None else None  # (..., 1) or None

        # 1. Write the directed edge on both pointer tensors
        next_at_u = (
            v_idx if active is None else torch.where(active, v_idx, self._next.gather(-1, u_idx))
        )
        prev_at_v = (
            u_idx if active is None else torch.where(active, u_idx, self._prev.gather(-1, v_idx))
        )
        self._next.scatter_(-1, u_idx, next_at_u)
        self._prev.scatter_(-1, v_idx, prev_at_v)

        # 2. Depot is not on any route; restore its pointers in case u or v was the depot
        self._next[..., 0] = self.UNASSIGNED
        self._prev[..., 0] = self.UNASSIGNED

        # 3. Customer-to-customer edges merge two routes: smaller head index wins
        is_merge = (u_idx > 0) & (v_idx > 0)
        if active is not None:
            is_merge = is_merge & active
        u_route = self._route_id.gather(-1, u_idx)
        v_route = self._route_id.gather(-1, v_idx)
        winner = torch.minimum(u_route, v_route)
        loser = torch.maximum(u_route, v_route)
        self._route_id = torch.where(
            is_merge & (self._route_id == loser),
            winner,
            self._route_id,
        )

    def add_edge(self, u: torch.Tensor, v: torch.Tensor) -> CVRPGraphSolution:
        solution = self.clone()
        solution.add_edge_(u, v)
        return solution

    def insert_(
        self,
        c: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Inserts customer c between predecessor u and successor v. In-place.

        Either of u, v may be the depot (0). The new route_id of c is inherited
        from whichever of v, u is a customer; if both are the depot, c becomes a
        new singleton route headed by itself.

        Args:
            c: Customer indices to insert, shape (...,). Values in 1..N.
            u: Predecessor indices, shape (...,). 0 = depot.
            v: Successor indices, shape (...,). 0 = depot.
            mask: Optional (...,) bool. If provided, only batches where the mask
                is True are updated; others keep their previous state.
        """
        c_idx = c.unsqueeze(-1)
        u_idx = u.unsqueeze(-1)
        v_idx = v.unsqueeze(-1)
        active = (
            mask.unsqueeze(-1) if mask is not None else torch.ones_like(c_idx, dtype=torch.bool)
        )

        # 1. Set c's own pointers.
        self._prev.scatter_(-1, c_idx, torch.where(active, u_idx, self._prev.gather(-1, c_idx)))
        self._next.scatter_(-1, c_idx, torch.where(active, v_idx, self._next.gather(-1, c_idx)))

        # 2. Overwrite next[u] with c when u is a customer (depot keeps UNASSIGNED).
        self._next.scatter_(
            -1, u_idx, torch.where((u_idx > 0) & active, c_idx, self._next.gather(-1, u_idx))
        )

        # 3. Overwrite prev[v] with c when v is a customer.
        self._prev.scatter_(
            -1, v_idx, torch.where((v_idx > 0) & active, c_idx, self._prev.gather(-1, v_idx))
        )

        # 4. Depot is not on any route; restore its pointers.
        self._next[..., 0] = self.UNASSIGNED
        self._prev[..., 0] = self.UNASSIGNED

        # 5. Inherit route id from v if customer, else from u if customer, else self.
        v_route = self._route_id.gather(-1, v_idx)
        u_route = self._route_id.gather(-1, u_idx)
        route_id_inherited = torch.where(v_idx > 0, v_route, torch.where(u_idx > 0, u_route, c_idx))
        self._route_id.scatter_(
            -1, c_idx, torch.where(active, route_id_inherited, self._route_id.gather(-1, c_idx))
        )

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
