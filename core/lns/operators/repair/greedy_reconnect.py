from __future__ import annotations

import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.cvrp.graph_solution import CVRPGraphSolution


class CVRPGreedyReconnectRepair(RepairOperator[CVRPGraphSolution]):
    """Greedy reconnection repair for CVRPGraphSolution.

    Each step connects one chain tail (next == UNASSIGNED) to the nearest valid
    free head (prev == UNASSIGNED), respecting capacity. If no feasible head
    exists, the tail is closed to the depot. After the loop, any remaining free
    heads are connected to the depot, completing all routes.
    """

    def __call__(self, partial_solution: CVRPGraphSolution) -> CVRPGraphSolution:
        solution = partial_solution.clone()
        E = int(solution.num_unassigned.max().item())

        for _ in range(E):
            is_active = solution.num_unassigned > 0  # (...,)
            if not is_active.any():
                break

            is_tail = solution._next[..., 1:] == solution.UNASSIGNED  # (..., N)
            is_free_head = solution._prev[..., 1:] == solution.UNASSIGNED  # (..., N)

            # Pick one tail per batch element. argmax returns 0 when all False (inactive),
            # giving tail_node = 1 - the dummy node used for the inactive no-op below.
            tail_node = is_tail.long().argmax(-1) + 1  # (...,) in 1..N
            tail_node = torch.where(is_active, tail_node, torch.ones_like(tail_node))

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
