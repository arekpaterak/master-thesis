from __future__ import annotations

import math

import torch

from core.lns.operators.repair import RepairOperator
from core.problems.tsp.graph_solution import TSPGraphSolution


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
