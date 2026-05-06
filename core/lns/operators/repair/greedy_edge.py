import math

import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.tsp.graph_solution import TSPGraphSolution


class TSPGreedyEdgeRepair(RepairOperator[TSPGraphSolution]):
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
        M = int(num_unassigned.max().item())

        for _ in range(M):
            active = num_unassigned > 0
            if not active.any():
                break

            unassigned_mask = next_nodes == UNASSIGNED
            start_node = unassigned_mask.float().argmax(dim=-1)
            potential_end_nodes = prev_nodes == UNASSIGNED

            needs_cycle_check = active & (num_unassigned > 1)
            if needs_cycle_check.any():
                path_start = self._find_path_starts(prev_nodes, start_node, UNASSIGNED, num_nodes)
                forbidden_mask = torch.zeros(
                    (flat_batch_size, num_nodes), device=device, dtype=torch.bool
                )
                forbidden_mask.scatter_(1, path_start.unsqueeze(-1), True)
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
    def _find_path_starts(
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
