from __future__ import annotations

import math

import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.tsp.graph_solution import TSPGraphSolution


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
        self._seed = seed
        self._deconstruction_size = deconstruction_size
        self._generator: torch.Generator | None = None

    def __call__(self, deconstructed_solution: CVRPGraphSolution) -> CVRPGraphSolution:
        solution = deconstructed_solution.clone()
        N = solution.instance.num_customers
        device = solution.instance.locations.device
        generator = self._device_generator(device)
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

    def _device_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            gen = torch.Generator(device=device)
            if self._seed is not None:
                gen.manual_seed(self._seed)
            self._generator = gen
        return self._generator

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


class TSPGreedyReconnectRepair(RepairOperator[TSPGraphSolution]):
    """Greedy reconnection repair for TSPGraphSolution.

    Pass ``deconstruction_size`` to use a static upper bound on the inner loop
    and skip the host-device sync that would otherwise be needed to size it.
    Removing M nodes from a tour creates at most 2M unassigned successors, so
    the bound is ``2 * M``.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        deconstruction_size: int | None = None,
    ) -> None:
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._deconstruction_size = deconstruction_size

    def __call__(self, deconstructed_solution: TSPGraphSolution) -> TSPGraphSolution:
        original_batch_size = deconstructed_solution.batch_size
        flat_batch_size = math.prod(original_batch_size) if original_batch_size else 1
        num_nodes = deconstructed_solution.instance.num_nodes
        device = deconstructed_solution.instance.device

        next_nodes = deconstructed_solution.next_nodes.reshape(flat_batch_size, num_nodes).clone()
        prev_nodes = deconstructed_solution.prev_nodes.reshape(flat_batch_size, num_nodes).clone()
        dist_matrix = deconstructed_solution.instance.distances().reshape(
            flat_batch_size, num_nodes, num_nodes
        )

        UNASSIGNED = deconstructed_solution.UNASSIGNED
        num_unassigned = (next_nodes == UNASSIGNED).sum(dim=-1)
        if self._deconstruction_size is None:
            E = int(num_unassigned.max().item())
        else:
            # See CVRP variant for the 2M reasoning. Iterations past the true
            # max are no-ops gated by the `active` mask below.
            E = min(2 * self._deconstruction_size, num_nodes)

        for _ in range(E):
            active = num_unassigned > 0

            unassigned_mask = next_nodes == UNASSIGNED
            start_node = self._select_start_node(unassigned_mask, device)
            potential_end_nodes = prev_nodes == UNASSIGNED

            needs_cycle_check = active & (num_unassigned > 1)
            if needs_cycle_check.any():
                fragment_start = self._find_fragment_start(
                    prev_nodes, start_node, UNASSIGNED, num_nodes
                )
                forbidden_mask = torch.zeros(
                    (flat_batch_size, num_nodes), device=device, dtype=torch.bool
                )
                forbidden_mask.scatter_(1, fragment_start.unsqueeze(-1), True)
                potential_end_nodes = potential_end_nodes & ~(
                    forbidden_mask & needs_cycle_check.unsqueeze(-1)
                )

            distances = dist_matrix.gather(
                1, start_node.view(-1, 1, 1).expand(-1, 1, num_nodes)
            ).squeeze(1)
            distances[~potential_end_nodes] = float("inf")
            selected_end_node = distances.argmin(dim=-1)

            active_col = active.unsqueeze(-1)
            start_col = start_node.unsqueeze(-1)
            end_col = selected_end_node.unsqueeze(-1)
            next_nodes.scatter_(
                -1, start_col, torch.where(active_col, end_col, next_nodes.gather(-1, start_col))
            )
            prev_nodes.scatter_(
                -1, end_col, torch.where(active_col, start_col, prev_nodes.gather(-1, end_col))
            )

            num_unassigned = (next_nodes == UNASSIGNED).sum(dim=-1)

        return TSPGraphSolution(
            _instance=deconstructed_solution.instance,
            _next=next_nodes.reshape(*original_batch_size, num_nodes),
            _prev=prev_nodes.reshape(*original_batch_size, num_nodes),
            batch_size=torch.Size(original_batch_size),
        )

    @staticmethod
    def _find_fragment_start(
        prev_nodes: torch.Tensor, current_nodes: torch.Tensor, unassigned: int, num_nodes: int
    ) -> torch.Tensor:
        current = current_nodes.clone()
        for _ in range(num_nodes):
            prev = prev_nodes.gather(-1, current.unsqueeze(-1)).squeeze(-1)
            mask = prev != unassigned
            if not mask.any():
                break
            current[mask] = prev[mask]
        return current

    def _select_start_node(
        self, unassigned_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        rand_scores = torch.rand(unassigned_mask.shape, generator=self._generator).to(device)
        return rand_scores.masked_fill(~unassigned_mask, -1.0).argmax(dim=-1)
