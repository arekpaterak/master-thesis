import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.solution import BaseSolution
from core.problems.tsp.solution import TSPSolution


class GreedyRepair[S: BaseSolution](RepairOperator[S]):
    def __call__(self, partial_solution: S) -> S:
        match partial_solution:
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
                raise NotImplementedError(f"Operator not implemented for {type(partial_solution)}")

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
