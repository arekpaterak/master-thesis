from __future__ import annotations

import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.random import SeededGenerator


class CVRPGreedyReconnectRepair(RepairOperator[CVRPGraphSolution]):
    """Greedy reconnection repair for CVRPGraphSolution.

    Each iteration picks one route tail (a customer with ``next == UNASSIGNED``)
    at random, then connects it to the nearest free head (``prev == UNASSIGNED``)
    that lives in a different route and would not push the merged route over
    capacity. If no such head exists, the tail is closed to the depot. After the
    loop, any still-unassigned ``_prev`` entries are clamped to the depot.

    Pass ``deconstruction_size`` to use a static upper bound on the inner loop
    and skip the host-device sync that would otherwise size it. Removing M
    customers creates at most 2M tails (M removed nodes plus up to M of their
    non-removed predecessors), so the bound is ``2 * M``.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        deconstruction_size: int | None = None,
    ) -> None:
        self._rng = SeededGenerator(seed)
        self._deconstruction_size = deconstruction_size

    def __call__(self, deconstructed_solution: CVRPGraphSolution) -> CVRPGraphSolution:
        solution = deconstructed_solution.clone()
        N = solution.instance.num_customers
        device = solution.instance.locations.device
        generator = self._rng.on(device)
        customer_distances = solution.instance.distances()[..., 1:, 1:]  # (..., N, N)

        for _ in range(self._loop_bound(solution, N)):
            is_tail_unassigned = solution._next[..., 1:] == solution.UNASSIGNED  # (..., N)
            is_free_head = solution._prev[..., 1:] == solution.UNASSIGNED  # (..., N)

            tail_node = self._pick_tail(is_tail_unassigned, generator, device)
            valid_head = self._valid_heads(solution, tail_node, is_free_head)
            v_node = self._nearest_or_depot(tail_node, valid_head, customer_distances)

            is_active = is_tail_unassigned.any(dim=-1)  # (...,)
            solution.add_edge_(tail_node, v_node, mask=is_active)

        # Close any remaining free heads (prev == UNASSIGNED) to the depot.
        solution._prev[..., 1:] = solution._prev[..., 1:].clamp(min=0)
        return solution

    def _loop_bound(self, solution: CVRPGraphSolution, N: int) -> int:
        if self._deconstruction_size is None:
            num_tails = (solution._next[..., 1:] == solution.UNASSIGNED).sum(dim=-1)
            return int(num_tails.max().item())
        # Iterations past the true max are no-ops gated by ``is_active`` on add_edge_.
        return min(2 * self._deconstruction_size, N)

    @staticmethod
    def _pick_tail(
        is_tail_unassigned: torch.Tensor,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Random tail per batch element. Returns (...,) in 1..N."""
        scores = torch.rand(is_tail_unassigned.shape, generator=generator, device=device)
        return scores.masked_fill(~is_tail_unassigned, -1.0).argmax(-1) + 1

    @staticmethod
    def _valid_heads(
        solution: CVRPGraphSolution,
        tail_node: torch.Tensor,
        is_free_head: torch.Tensor,
    ) -> torch.Tensor:
        """Candidate heads that are free, in a different route, and capacity-feasible."""
        route_demands = solution.demand_per_route()  # (..., max_route_id+1)
        tail_route_id = solution._route_id.gather(-1, tail_node[..., None]).squeeze(-1)
        tail_route_demand = route_demands.gather(-1, tail_route_id[..., None]).squeeze(-1)

        head_route_ids = solution._route_id[..., 1:]  # (..., N)
        head_route_demand = route_demands.gather(-1, head_route_ids)
        capacity_ok = tail_route_demand[..., None] + head_route_demand <= solution.instance.capacity
        different_route = head_route_ids != tail_route_id[..., None]
        return is_free_head & capacity_ok & different_route

    @staticmethod
    def _nearest_or_depot(
        tail_node: torch.Tensor,
        valid_head: torch.Tensor,
        customer_distances: torch.Tensor,
    ) -> torch.Tensor:
        """Pick nearest valid head; fall back to depot (0) if none exists."""
        N = valid_head.shape[-1]
        tail_idx = (tail_node - 1)[..., None, None].expand(*tail_node.shape, 1, N)
        row = customer_distances.gather(-2, tail_idx).squeeze(-2)
        row = row.masked_fill(~valid_head, float("inf"))
        best_head_node = row.argmin(-1) + 1  # 1..N
        has_valid_head = valid_head.any(-1)
        return torch.where(has_valid_head, best_head_node, torch.zeros_like(best_head_node))
