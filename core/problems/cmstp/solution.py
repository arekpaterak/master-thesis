from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from core.constants import EPS
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.solution import BaseSolution

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes


class CMSTPSolution(BaseSolution):
    """Parent-pointer representation of a capacitated minimum spanning tree.

    A CVRP route is a degenerate max-degree-2 tree; relaxing that fan-out to any
    degree gives a CMSTP. Where CVRP tracks ``next``/``prev``/``route_id``, a tree
    rooted at node 0 needs only one parent pointer per node plus the subtree it
    hangs off. ``subtree_id`` is the CMSTP analog of CVRP's ``route_id``: the
    direct-child-of-root ancestor a node belongs to, and the unit capacity bounds.

    All fields are (..., N+1) with node 0 the root. Terminals are 1..N.
    """

    UNASSIGNED = -1
    ROOT = 0

    _instance: CMSTPInstance
    _parent: torch.Tensor
    """(..., N+1) parent per node. 0 = root, 1..N = terminal, -1 = unassigned.

    Root (index 0) is always -1. A terminal whose parent is 0 is a direct child of
    the root (a subtree head); a terminal with parent -1 is a detached fragment.
    """
    _subtree_id: torch.Tensor
    """(..., N+1) which root subtree each node belongs to.

    Always -1 for the root. For terminals, equals the index of the direct-child-of-root
    ancestor (the subtree head). A detached/isolated terminal is its own head, so its
    subtree_id equals its own index -- mirrors CVRP's route_id fragment convention.
    """

    @classmethod
    def star(cls, instance: CMSTPInstance) -> CMSTPSolution:
        """Trivial feasible-shaped start: every terminal is a direct child of the root.

        Each terminal is its own subtree, so this is feasible iff each single demand
        is within capacity (always true for standard samplers). The CMSTP analog of
        CVRP's ``one_customer_per_route``.
        """
        N = instance.num_terminals
        batch_size = instance.batch_size
        device = instance.locations.device

        _parent = torch.full((*batch_size, N + 1), cls.ROOT, dtype=torch.long, device=device)
        _parent[..., 0] = cls.UNASSIGNED

        _subtree_id = (
            torch.arange(N + 1, dtype=torch.long, device=device).expand(*batch_size, N + 1).clone()
        )
        _subtree_id[..., 0] = cls.UNASSIGNED

        return cls(
            _instance=instance,
            _parent=_parent,
            _subtree_id=_subtree_id,
            batch_size=batch_size,
        )

    @staticmethod
    def _compute_subtree_ids(parent: torch.Tensor, num_terminals: int) -> torch.Tensor:
        """Recompute subtree IDs from scratch (torch, GPU-safe).

        Args:
            parent: Parent pointers of shape (..., N+1).
            num_terminals: N, excluding the root.

        Returns:
            Subtree ID tensor of shape (..., N+1). Only the root (index 0) is -1.
        """
        V = num_terminals + 1
        device = parent.device
        batch_shape = parent.shape[:-1]
        arange = torch.arange(V, device=device).expand(*batch_shape, V)

        # parent > 0: follow the parent. parent <= 0 (root child or detached): self-loop.
        head = torch.where(parent > 0, parent, arange)

        for _ in range(math.ceil(math.log2(V + 1))):
            head = head.gather(-1, head)

        subtree_id = head.clone()
        subtree_id[..., 0] = CMSTPSolution.UNASSIGNED
        return subtree_id

    def __post_init__(self) -> None:
        assert self._parent.shape == self._subtree_id.shape

        N = self.instance.num_terminals
        assert ((self._parent >= -1) & (self._parent <= N)).all(), "Invalid node index in _parent"

        assert (self._subtree_id[..., 1:] >= 0).all(), "Terminal subtree IDs must be >= 0"
        assert (self._subtree_id[..., 0] == self.UNASSIGNED).all(), "Root subtree ID must be -1"
        assert (self._parent[..., 0] == self.UNASSIGNED).all(), "Root parent should be unassigned"

    @property
    def instance(self) -> CMSTPInstance:
        return self._instance

    @property
    def parent_nodes(self) -> torch.Tensor:
        return self._parent

    @property
    def subtree_ids(self) -> torch.Tensor:
        return self._subtree_id

    def parent_per_terminal(self) -> torch.Tensor:
        """(..., N) int - parent node index per terminal. 0 = root, 1..N = terminal."""
        return self._parent[..., 1:]

    def subtree_per_terminal(self) -> torch.Tensor:
        """(..., N) int - subtree (root-branch) ID per terminal. Always >= 1."""
        return self._subtree_id[..., 1:]

    @property
    def is_complete(self) -> torch.Tensor:
        return self.num_unassigned == 0

    @property
    def is_feasible(self) -> torch.Tensor:
        """(...) bool - complete AND every root subtree is within capacity."""
        subtree_demands = self.demand_per_subtree()
        return self.is_complete & (subtree_demands <= self._instance.capacity + EPS).all(dim=-1)

    @property
    def num_unassigned(self) -> torch.Tensor:
        """(...) int - number of terminals without a parent."""
        return (self._parent[..., 1:] == self.UNASSIGNED).sum(dim=-1)

    def node_degree(self) -> torch.Tensor:
        """(..., N) int - tree degree per terminal: parent edge plus child edges.

        Unlike CVRP (bounded at 2), a tree node's degree is unbounded, so children are
        counted with a scatter-add over the parent array rather than a single ``has_next``.
        """
        N = self._instance.num_terminals
        has_parent = (self._parent[..., 1:] != self.UNASSIGNED).long()  # (..., N)

        safe_parent = self._parent.clamp(min=0)  # (..., N+1); detached -> 0, ignored below
        child_count = torch.zeros(
            *self._parent.shape[:-1], N + 1, dtype=torch.long, device=self._parent.device
        )
        is_edge = (self._parent != self.UNASSIGNED).long()  # exclude root and detached
        child_count.scatter_add_(-1, safe_parent, is_edge)
        return has_parent + child_count[..., 1:]

    def demand_per_subtree(self) -> torch.Tensor:
        """(..., N+1) float - total demand accumulated per root subtree.

        Subtree IDs are head-node indices (sparse in 1..N). Sized statically to N+1 to
        avoid a host sync on max(subtree_id); unused indices stay at 0. The CMSTP analog
        of CVRP ``demand_per_route``.
        """
        subtree_ids = self._subtree_id[..., 1:]  # (..., N), all >= 1
        demands = self._instance.demand[..., 1:]  # (..., N)
        N = self._instance.num_terminals
        out = torch.zeros(
            *subtree_ids.shape[:-1], N + 1, dtype=demands.dtype, device=demands.device
        )
        out.scatter_add_(-1, subtree_ids, demands)
        return out

    def total_distance(self, *, rounded: bool = False) -> torch.Tensor:
        """(...) float - total length of all tree edges (each node to its parent).

        A tree over N+1 nodes has exactly N edges; summing each terminal's edge to its
        parent counts every edge once (the root has no parent edge).
        """
        assert self.is_complete.all(), (
            "total_distance() requires all terminals to be assigned; "
            f"found {self.num_unassigned.max().item()} unassigned in at least one batch element"
        )
        locations = self._instance.locations  # (..., N+1, 2)
        if rounded:
            locations = locations * self._instance.coordinate_scale.unsqueeze(-1)

        terminal_locs = locations[..., 1:, :]  # (..., N, 2)
        parent = self._parent[..., 1:]  # (..., N), all >= 0 when complete
        parent_locs = locations.gather(-2, parent.unsqueeze(-1).expand(*parent.shape, 2))
        edge_dists = (terminal_locs - parent_locs).pow(2).sum(-1).sqrt()  # (..., N)
        if rounded:
            edge_dists = edge_dists.round()
        return edge_dists.sum(-1)

    def objective(self) -> torch.Tensor:
        return self.total_distance()

    # --- numpy bridge (encapsulation boundary for the numba apply-and-repair) ------

    def to_numpy_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract contiguous numpy arrays for the numba apply-and-repair.

        Batch dims are collapsed to a single leading dim G; floats are float64. Pair with
        :meth:`from_numpy_arrays` to rebuild a solution from the numba output.

        Returns:
            ``(parent, locations, demand, capacity)`` with shapes ``(G, N+1)`` int64,
            ``(G, N+1, 2)`` float64, ``(G, N+1)`` float64 and ``(G,)`` float64.
        """
        N = self._instance.num_terminals
        instance = self._instance
        parent = self._parent.reshape(-1, N + 1).detach().cpu().contiguous().numpy()
        locations = (
            instance.locations.reshape(-1, N + 1, 2).detach().cpu().double().contiguous().numpy()
        )
        demand = instance.demand.reshape(-1, N + 1).detach().cpu().double().contiguous().numpy()
        capacity = instance.capacity.reshape(-1).detach().cpu().double().contiguous().numpy()
        return parent, locations, demand, capacity

    @classmethod
    def from_numpy_arrays(
        cls,
        *,
        instance: CMSTPInstance,
        parent: np.ndarray,
        batch_size: torch.Size,
        device: torch.device,
    ) -> CMSTPSolution:
        """Rebuild a solution from numba-produced ``parent`` numpy arrays.

        ``subtree_id`` is recomputed from ``parent`` so the result satisfies the class
        invariants without the numba pass having to maintain it.
        """
        num_terminals = instance.num_terminals
        parent_tensor = (
            torch.from_numpy(parent)
            .to(device=device, dtype=torch.long)
            .reshape(*batch_size, num_terminals + 1)
        )
        subtree_id = cls._compute_subtree_ids(parent_tensor, num_terminals)
        return cls(
            _instance=instance,
            _parent=parent_tensor,
            _subtree_id=subtree_id,
            batch_size=batch_size,
        )

    def apply_and_repair(
        self,
        nodes: np.ndarray,
        *,
        n_repairs: int,
        base_seed: int,
        first_repair_uses_order: bool = False,
    ) -> CMSTPSolution:
        """Deconstruct by each rollout's node set and greedily repair, returning candidates.

        Wraps the fused numba parallel pass: every rollout is applied independently from
        this solution, with no acceptance. ``nodes`` is ``(G, K, M)`` int64 (terminals to
        remove per rollout). Returns a ``(*batch, K)`` solution.
        """
        from core.problems.cmstp._numba import apply_and_repair_parallel

        parent_np, locations, demand, capacity = self.to_numpy_arrays()
        parent_out, _ = apply_and_repair_parallel(
            parent=parent_np,
            nodes=nodes,
            locations=locations,
            demand=demand,
            capacity=capacity,
            n_repairs=n_repairs,
            base_seed=base_seed,
            first_repair_uses_order=first_repair_uses_order,
        )
        num_rollouts = parent_out.shape[1]
        instance = self._instance.reshape(*self.batch_size, 1).expand(
            *self.batch_size, num_rollouts
        )
        return type(self).from_numpy_arrays(
            instance=instance,
            parent=parent_out,
            batch_size=torch.Size([*self.batch_size, num_rollouts]),
            device=self.device,
        )

    def apply_and_repair_sequential(
        self,
        nodes_to_remove: np.ndarray,
        *,
        n_repairs: int,
        accept_kind: int,
        temperature: float,
        eps: float,
        base_seed: int,
        first_repair_uses_order: bool = False,
    ) -> CMSTPSolution:
        """Apply each element's rollouts sequentially with acceptance, threading the result.

        Wraps the fused numba sequential pass. ``nodes_to_remove`` is ``(G, R, M)`` int64.
        Returns a solution with this solution's batch shape.
        """
        from core.problems.cmstp._numba import apply_and_repair_sequential

        parent_np, locations, demand, capacity = self.to_numpy_arrays()
        parent_out = apply_and_repair_sequential(
            parent=parent_np,
            nodes_to_remove=nodes_to_remove,
            locations=locations,
            demand=demand,
            capacity=capacity,
            n_repairs=n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=eps,
            base_seed=base_seed,
            first_repair_uses_order=first_repair_uses_order,
        )
        return type(self).from_numpy_arrays(
            instance=self._instance,
            parent=parent_out,
            batch_size=self.batch_size,
            device=self.device,
        )

    def visualize(
        self,
        *,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
        with_problem_size: bool = True,
        with_objective: bool = True,
        separator: str = " | ",
        palette: str = "tab20",
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
            separator=separator,
            palette=palette,
            scale=scale,
        )
