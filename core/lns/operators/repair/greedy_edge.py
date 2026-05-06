import torch

from core.lns.operators.repair.base import RepairOperator
from core.problems.tsp.graph_solution import TSPGraphSolution


class TSPGreedyEdgeRepair(RepairOperator[TSPGraphSolution]):
    def __call__(self, partial_solution: TSPGraphSolution) -> TSPGraphSolution:
        s = partial_solution.clone()

        batch_size = s.batch_size[0]
        num_nodes = s.instance.num_nodes
        device = s.instance.device

        dist_matrix = s.instance.distances()

        M = int(s.num_unassigned[0].item())

        for i in range(M):
            unassigned_mask = s.next_nodes == s.UNASSIGNED
            start_node = unassigned_mask.float().argmax(dim=-1)
            potential_end_nodes = s.prev_nodes == s.UNASSIGNED

            # We shouldn't connect nodes if they are in the same route,
            # UNLESS it's the very last edge to be added.
            if i < M - 1:
                path_start = self._find_path_starts(s, start_node)
                forbidden_mask = torch.zeros(
                    (batch_size, num_nodes), device=device, dtype=torch.bool
                )
                forbidden_mask.scatter_(1, path_start.unsqueeze(-1), True)
                potential_end_nodes = potential_end_nodes & ~forbidden_mask

            distances = dist_matrix.gather(
                1, start_node.view(-1, 1, 1).expand(-1, 1, num_nodes)
            ).squeeze(1)

            distances[~potential_end_nodes] = float("inf")

            selected_end_node = distances.argmin(dim=-1)

            new_edges = torch.stack([start_node, selected_end_node], dim=-1).unsqueeze(1)
            s.add_edges(new_edges)

        return s

    def _find_path_starts(
        self, solution: TSPGraphSolution, current_nodes: torch.Tensor
    ) -> torch.Tensor:
        current = current_nodes.clone()
        for _ in range(solution.instance.num_nodes):
            prev = solution.prev_nodes.gather(-1, current.unsqueeze(-1)).squeeze(-1)
            mask = prev != solution.UNASSIGNED
            if not mask.any():
                break
            current[mask] = prev[mask]
        return current
