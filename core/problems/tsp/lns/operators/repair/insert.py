from __future__ import annotations

import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.solution import BaseSolution
from core.problems.tsp import TSPSolution
from core.problems.tsp.graph_solution import TSPGraphSolution


class TSPGreedyInsertRepair[S: BaseSolution](RepairOperator[S]):
    repair_type = "greedy_insert"
    problem_name = "tsp"
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

    def __call__(self, deconstructed_solution: S) -> S:
        match deconstructed_solution:
            case TSPGraphSolution() as s:
                return self._call_graph(s)
            case TSPSolution() as s:
                return self._call_sequence(s)
            case _:
                raise NotImplementedError(
                    f"Operator not implemented for {type(deconstructed_solution)}"
                )

    def _device_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            gen = torch.Generator(device=device)
            if self._seed is not None:
                gen.manual_seed(self._seed)
            self._generator = gen
        return self._generator

    def _loop_bound(self, solution: TSPGraphSolution) -> int:
        if self._deconstruction_size is None:
            return int(solution.num_unassigned.max().item())
        return min(self._deconstruction_size, solution.instance.num_nodes)

    @staticmethod
    def _pick_unassigned_node(
        is_free: torch.Tensor,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Random free node per batch. Returns (...,)."""
        scores = torch.rand(is_free.shape, generator=generator, device=device)
        return scores.masked_fill(~is_free, -1.0).argmax(-1)

    def _call_graph(self, deconstructed_solution: TSPGraphSolution) -> TSPGraphSolution:
        solution = deconstructed_solution.clone()
        device = solution.instance.locations.device
        generator = self._device_generator(device)

        distances = solution.instance.distances()  # constant across iterations
        for _ in range(self._loop_bound(solution)):
            is_unassigned = (solution.prev_nodes == solution.UNASSIGNED) & (
                solution.next_nodes == solution.UNASSIGNED
            )  # (..., N)

            node = self._pick_unassigned_node(is_unassigned, generator, device)
            u_positions, v_positions, costs = self._enumerate_insertion_positions(
                solution, node, distances
            )
            best_position = costs.argmin(-1)
            chosen_u = u_positions.gather(-1, best_position[..., None]).squeeze(-1)
            chosen_v = v_positions.gather(-1, best_position[..., None]).squeeze(-1)

            is_active = is_unassigned.any(dim=-1)  # (...,)
            solution.insert_(node, chosen_u, chosen_v, mask=is_active)

        return solution

    @staticmethod
    def _enumerate_insertion_positions(
        solution: TSPGraphSolution,
        node: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialise candidate insertion positions as (u, v, cost) tensors.

        Each position corresponds to a directed edge (u, v) in the current solution.
        Inserting c splits the edge: u -> c -> v. Invalid positions get cost = +inf.
        """
        n = solution.instance.num_nodes
        batch_shape = node.shape

        distance_from_node = distances.gather(
            -2, node[..., None, None].expand(*batch_shape, 1, n)
        ).squeeze(-2)  # (..., N)
        flat_dist = distances.flatten(-2, -1)  # (..., N^2)

        u, v, valid = solution.insertion_positions()  # (..., N) each

        costs = (
            distance_from_node.gather(-1, u)
            + distance_from_node.gather(-1, v)
            - flat_dist.gather(-1, u * n + v)
        )
        costs.masked_fill_(~valid, float("inf"))
        return u, v, costs

    def _call_sequence(self, s: TSPSolution) -> TSPSolution:
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
