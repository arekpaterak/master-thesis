from __future__ import annotations

import math

import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.tsp.graph_solution import TSPGraphSolution


class CVRPGreedyReconnectRepair(RepairOperator[CVRPGraphSolution]):
    """Greedy reconnection repair for CVRPGraphSolution.

    In each step, it connects one chain tail (next == UNASSIGNED) to the nearest valid free head (prev == UNASSIGNED), respecting capacity.
    If no feasible head exists, the tail is closed to the depot.
    After the loop, any remaining free heads are connected to the depot, completing all routes.

    Pass ``deconstruction_size`` to use a static upper bound on the inner loop
    and skip the host-device sync that would otherwise be needed to size it.
    Removing M customers creates at most 2M tails (M removed nodes plus up to
    M of their non-removed predecessors), so the bound is ``2 * M``.
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

    def __call__(self, partial_solution: CVRPGraphSolution) -> CVRPGraphSolution:
        solution = partial_solution.clone()
        N = solution.instance.num_customers
        if self._deconstruction_size is None:
            num_tails = (solution._next[..., 1:] == solution.UNASSIGNED).sum(dim=-1)
            E = int(num_tails.max().item())
        else:
            # Iterations past the true max are no-ops gated by `is_active`.
            E = min(2 * self._deconstruction_size, N)

        device = solution.instance.locations.device
        for _ in range(E):
            is_active = (solution._next[..., 1:] == solution.UNASSIGNED).any(dim=-1)  # (...,)

            is_tail = solution._next[..., 1:] == solution.UNASSIGNED  # (..., N)
            is_free_head = solution._prev[..., 1:] == solution.UNASSIGNED  # (..., N)

            # Pick one tail per batch element.
            # When inactive, argmax returns 0 giving tail_node = 1 - the dummy node used for the no-op below.
            tail_node = self._select_tail(is_tail, is_active, device)  # (...,) in 1..N

            # Capacity feasibility per candidate head
            route_demands = solution.demand_per_route()  # (..., max_id+1)
            tail_route_ids = solution._route_id.gather(-1, tail_node.unsqueeze(-1)).squeeze(-1)
            tail_route_demand = route_demands.gather(-1, tail_route_ids.unsqueeze(-1)).squeeze(-1)
            head_route_ids = solution._route_id[..., 1:]  # (..., N)
            head_route_demands = route_demands.gather(-1, head_route_ids)
            capacity_ok = (
                tail_route_demand.unsqueeze(-1) + head_route_demands
            ) <= solution.instance.capacity

            # Cycle prevention: cannot connect tail to a head in its own chain
            same_chain = head_route_ids == tail_route_ids.unsqueeze(-1)  # (..., N)

            valid_head = is_free_head & capacity_ok & ~same_chain  # (..., N)

            # Greedy nearest valid head
            locations = solution.instance.locations  # (..., N+1, 2)
            tail_locs = locations.gather(
                -2, tail_node.unsqueeze(-1).unsqueeze(-1).expand(*tail_node.shape, 1, 2)
            ).squeeze(-2)  # (..., 2)
            distances = (tail_locs.unsqueeze(-2) - locations[..., 1:, :]).pow(2).sum(-1).sqrt()
            distances = distances.masked_fill(~valid_head, float("inf"))

            has_valid_head = valid_head.any(-1)  # (...,)
            best_head_node = distances.argmin(-1) + 1  # (...,) in 1..N

            # For inactive batch elements: re-write node 1's existing successor (no-op).
            existing_next_1 = solution._next[..., 1]
            v_node = torch.where(
                is_active & has_valid_head,
                best_head_node,
                torch.where(is_active, torch.zeros_like(tail_node), existing_next_1),
            )

            solution.add_edge_(tail_node, v_node)

        # Close any remaining free heads (prev == UNASSIGNED) to the depot.
        solution._prev[..., 1:] = solution._prev[..., 1:].clamp(min=0)
        return solution

    def _select_tail(
        self,
        is_tail: torch.Tensor,
        is_active: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        rand_scores = torch.rand(is_tail.shape, generator=self._generator).to(device)
        tail_node = rand_scores.masked_fill(~is_tail, -1.0).argmax(-1) + 1
        return torch.where(is_active, tail_node, torch.ones_like(tail_node))


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

    def __call__(self, partial_solution: TSPGraphSolution) -> TSPGraphSolution:
        original_batch_size = partial_solution.batch_size
        flat_batch_size = math.prod(original_batch_size) if original_batch_size else 1
        num_nodes = partial_solution.instance.num_nodes
        device = partial_solution.instance.device

        next_nodes = partial_solution.next_nodes.reshape(flat_batch_size, num_nodes).clone()
        prev_nodes = partial_solution.prev_nodes.reshape(flat_batch_size, num_nodes).clone()
        dist_matrix = partial_solution.instance.distances().reshape(
            flat_batch_size, num_nodes, num_nodes
        )

        UNASSIGNED = partial_solution.UNASSIGNED
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
            _instance=partial_solution.instance,
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
