import math

import torch

from core.constants import EPS
from core.lns.operators.repair.base import RepairOperator
from core.problems.cvrp.solution import CVRPSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.solution import TSPSolution


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


class CVRPGreedyInsertRepair(RepairOperator[CVRPSolution]):
    """
    Inherently sequential.
    Repairs a single partial solution from a batch at a time.
    Can insert inside the existing routes.
    """

    def __init__(self, *, seed: int | None = None):
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(self, deconstructed_solution: CVRPSolution) -> CVRPSolution:
        batch_size = deconstructed_solution.batch_size
        num_customers = deconstructed_solution.instance.num_customers
        max_len = 2 * num_customers + 1
        flat_batch_size = math.prod(batch_size) if batch_size else 1

        sequence = deconstructed_solution.sequence.reshape(flat_batch_size, -1).cpu()
        locations = (
            deconstructed_solution.instance.locations.reshape(flat_batch_size, -1, 2).float().cpu()
        )
        demands = deconstructed_solution.instance.demand.reshape(flat_batch_size, -1).float().cpu()
        capacity = deconstructed_solution.instance.capacity.reshape(flat_batch_size).float().cpu()

        repaired = [
            self._repair_single(
                sequence=sequence[b],
                locations=locations[b],
                demands=demands[b],
                capacity=float(capacity[b]),
                num_customers=num_customers,
                max_len=max_len,
            )
            for b in range(flat_batch_size)
        ]
        new_sequence = (
            torch.stack(repaired)
            .to(
                device=deconstructed_solution.sequence.device,
                dtype=deconstructed_solution.sequence.dtype,
            )
            .reshape(*batch_size, max_len)
        )
        return CVRPSolution.create(instance=deconstructed_solution.instance, sequence=new_sequence)

    def _repair_single(
        self,
        *,
        sequence: torch.Tensor,
        locations: torch.Tensor,
        demands: torch.Tensor,
        capacity: float,
        num_customers: int,
        max_len: int,
    ) -> torch.Tensor:
        routes: list[list[int]] = []
        current: list[int] = []
        for node in sequence.tolist():
            if node == 0:
                if current:
                    routes.append(current)
                current = []
            elif node >= 1:
                current.append(node)
        if current:
            routes.append(current)

        assigned = {c for route in routes for c in route}
        missing_customers = [c for c in range(1, num_customers + 1) if c not in assigned]

        if not missing_customers:
            tokens = [x for x in sequence.tolist() if x != -1]
            while len(tokens) < max_len:
                tokens.append(0)
            return torch.tensor(tokens[:max_len], dtype=sequence.dtype)

        perm = torch.randperm(len(missing_customers), generator=self._generator).tolist()
        missing_customers = [missing_customers[i] for i in perm]

        dist_mat = torch.cdist(locations.unsqueeze(0), locations.unsqueeze(0)).squeeze(0).tolist()
        route_demands = [sum(demands[c].item() for c in r) for r in routes]

        for customer in missing_customers:
            customer_demand = demands[customer].item()
            best_cost, best_r, best_pos = float("inf"), -1, -1
            for r_idx, route in enumerate(routes):
                if route_demands[r_idx] + customer_demand > capacity + EPS:
                    continue
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    next = route[pos] if pos < len(route) else 0
                    cost = (
                        dist_mat[prev][customer] + dist_mat[customer][next] - dist_mat[prev][next]
                    )
                    if cost < best_cost:
                        best_cost, best_r, best_pos = cost, r_idx, pos
            if best_r >= 0:
                routes[best_r].insert(best_pos, customer)
                route_demands[best_r] += customer_demand
            else:
                routes.append([customer])
                route_demands.append(customer_demand)

        tokens = [0]
        for route in routes:
            tokens.extend(route)
            tokens.append(0)
        while len(tokens) < max_len:
            tokens.append(0)
        return torch.tensor(tokens[:max_len], dtype=sequence.dtype)
