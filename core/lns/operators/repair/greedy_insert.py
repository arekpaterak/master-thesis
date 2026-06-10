from __future__ import annotations

import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.solution import TSPSolution


# TODO: Rewrite as batched, similar to CVRP
class TSPGreedyInsertRepair[S: BaseSolution](RepairOperator[S]):
    def __call__(self, deconstructed_solution: S) -> S:
        match deconstructed_solution:
            case TSPSolution() as s:
                if not (s.tour == s.UNASSIGNED).any():
                    return s

                # Reshape to (Batch, N) for unified processing
                original_shape = s.tour.shape
                tour = s.tour.view(-1, original_shape[-1])
                locs = s.instance.locations.view(-1, original_shape[-1], 2)

                # Precompute all-to-all distances for the batch: (B, N, N)
                dist_mat = torch.cdist(locs, locs)

                repaired_tours = []
                for b in range(tour.shape[0]):
                    repaired_tours.append(self._repair_single(tour[b], dist_mat[b], s.UNASSIGNED))

                repaired = torch.stack(repaired_tours, dim=0).view(original_shape)
                return TSPSolution.create(instance=s.instance, tour=repaired)

            case _:
                raise NotImplementedError(
                    f"Operator not implemented for {type(deconstructed_solution)}"
                )

    @staticmethod
    def _repair_single(tour: torch.Tensor, dists: torch.Tensor, unassigned: int) -> torch.Tensor:
        n = tour.shape[0]
        tour = tour.clone()

        # Handle empty tours
        if (tour != unassigned).sum() == 0:
            return torch.arange(n, device=tour.device, dtype=tour.dtype)

        while (tour == unassigned).any():
            # 1. Identify slots to fill and nodes available
            slots = (tour == unassigned).nonzero().flatten()
            assigned_mask = tour != unassigned
            all_nodes = torch.arange(n, device=tour.device, dtype=tour.dtype)

            # Find nodes not yet in the tour
            is_unassigned_node = torch.ones(n, dtype=torch.bool, device=tour.device)
            is_unassigned_node[tour[assigned_mask]] = False
            available_nodes = all_nodes[is_unassigned_node]

            # 2. Find neighbours for all slots simultaneously
            # (Finding the closest assigned index to the left/right circularly)
            assigned_idx = assigned_mask.nonzero().flatten()

            best_cost = float("inf")
            best_slot = None
            best_node = None

            for slot in slots:
                # Find preceding and succeeding assigned nodes in the tour
                # Logic: Find the assigned index immediately before/after this slot
                prev_pos = assigned_idx[assigned_idx < slot]
                prev_idx = prev_pos[-1] if prev_pos.numel() > 0 else assigned_idx[-1]

                next_pos = assigned_idx[assigned_idx > slot]
                next_idx = next_pos[0] if next_pos.numel() > 0 else assigned_idx[0]

                u = tour[prev_idx]
                v = tour[next_idx]

                # 3. Vectorized Cheapest Insertion: Cost = d(u, k) + d(k, v) - d(u, v)
                # available_nodes is a vector, dists[u, available_nodes] is a vector
                costs = dists[u, available_nodes] + dists[available_nodes, v] - dists[u, v]

                min_idx = torch.argmin(costs)
                min_val = costs[min_idx]

                if min_val < best_cost:
                    best_cost = min_val.item()
                    best_slot = slot
                    best_node = available_nodes[min_idx]

            tour[best_slot] = best_node

        return tour


class CVRPGreedyInsertRepair(RepairOperator[CVRPGraphSolution]):
    """Greedy cheapest-insertion repair for stitched CVRPGraphSolution.

    Each iteration picks one removed customer (``prev`` and ``next`` both
    UNASSIGNED) at random per batch element, then inserts it at the cheapest
    valid position across all current edges in the solution. Positions are every
    edge of every depot->...->depot route plus a new-route fallback
    (depot->c->depot). Capacity is enforced per route.

    Expects a stitched deconstructed solution (no fragmented routes). Pair
    with ``RandomNodeDestroy(..., stitch=True)``.

    Pass ``deconstruction_size`` to use a static upper bound on the inner loop
    and skip the host-device sync that would otherwise size it. The bound is
    M, since one customer is placed per iteration.
    """

    repair_type = "greedy_insert"
    problem_name = "cvrp"
    requires_stitch = True

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
        device = solution.instance.locations.device
        generator = self._device_generator(device)

        distances = solution.instance.distances()  # we cconstant across iterations
        for _ in range(self._loop_bound(solution)):
            is_unassigned = (solution._prev[..., 1:] == solution.UNASSIGNED) & (
                solution._next[..., 1:] == solution.UNASSIGNED
            )  # (..., N)

            customer = self._pick_unassigned_customer(is_unassigned, generator, device)
            u_positions, v_positions, costs = self._enumerate_insertion_positions(
                solution,
                customer,
                distances,
            )
            best_position = costs.argmin(-1)
            chosen_u = u_positions.gather(-1, best_position[..., None]).squeeze(-1)
            chosen_v = v_positions.gather(-1, best_position[..., None]).squeeze(-1)

            is_active = is_unassigned.any(dim=-1)  # (...,)
            solution.insert_(customer, chosen_u, chosen_v, mask=is_active)

        return solution

    def _device_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            gen = torch.Generator(device=device)
            if self._seed is not None:
                gen.manual_seed(self._seed)
            self._generator = gen
        return self._generator

    def _loop_bound(self, solution: CVRPGraphSolution) -> int:
        if self._deconstruction_size is None:
            is_free = (solution._prev[..., 1:] == solution.UNASSIGNED) & (
                solution._next[..., 1:] == solution.UNASSIGNED
            )
            return int(is_free.sum(dim=-1).max().item())
        return min(self._deconstruction_size, solution.instance.num_customers)

    @staticmethod
    def _pick_unassigned_customer(
        is_free: torch.Tensor,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Random free customer per batch. Returns (...,) in 1..N."""
        scores = torch.rand(is_free.shape, generator=generator, device=device)
        return scores.masked_fill(~is_free, -1.0).argmax(-1) + 1

    @staticmethod
    def _enumerate_insertion_positions(
        solution: CVRPGraphSolution,
        customer: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Materialise candidate insertion positions as (u, v, cost) tensors.

        Each position corresponds to a directed edge (u, v) in the current solution
        (or the synthetic new-route edge). Inserting c splits the edge: u -> c -> v.
        Invalid and capacity-infeasible positions get cost = +inf.
        The synthetic new-route edge guarantees argmin is always well-defined.
        """
        N = solution.instance.num_customers
        batch_shape = customer.shape
        demand = solution.instance.demand  # (..., N+1)
        capacity = solution.instance.capacity  # (...,)

        distance_from_customer = distances.gather(
            -2, customer[..., None, None].expand(*batch_shape, 1, N + 1)
        ).squeeze(-2)  # (..., N+1)
        flat_dist = distances.flatten(-2, -1)  # (..., (N+1)^2)

        demand_of_customer = demand.gather(-1, customer[..., None]).squeeze(-1)  # (...,)
        route_demands = solution.demand_per_route()  # (..., N+1)

        u, v, position_valid = solution.insertion_positions()  # (..., 2N+1) each

        # Insertion cost: d[u,c] + d[c,v] - d[u,v]
        costs = (
            distance_from_customer.gather(-1, u)
            + distance_from_customer.gather(-1, v)
            - flat_dist.gather(-1, u * (N + 1) + v)
        )

        # Capacity: customer endpoint of the edge is the route representative
        # (v when v is a customer; u for tail->depot and new-route where v == depot)
        route_owner = torch.where(v > 0, v, u)
        route_ids_at_owner = solution._route_id.gather(-1, route_owner)
        route_demand = route_demands.gather(-1, route_ids_at_owner.clamp(min=0))
        valid = position_valid & (route_demand + demand_of_customer.unsqueeze(-1) <= capacity)

        costs.masked_fill_(~valid, float("inf"))
        return u, v, costs
