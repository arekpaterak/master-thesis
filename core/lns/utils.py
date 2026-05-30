from __future__ import annotations

import torch

from core.problems.cvrp import CVRPSolution
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.solution import TSPSolution


def select_solution(mask: torch.Tensor, a: BaseSolution, b: BaseSolution) -> BaseSolution:
    """Select a[i] where mask[i] is True, else b[i]."""
    match a:
        case TSPSolution() as a_tsp_seq:
            assert isinstance(b, TSPSolution)
            return TSPSolution.create(
                instance=a_tsp_seq.instance,
                tour=torch.where(mask.unsqueeze(-1), a_tsp_seq.tour, b.tour),
            )
        case CVRPSolution() as a_cvrp:
            assert isinstance(b, CVRPSolution)
            return CVRPSolution.create(
                instance=a_cvrp.instance,
                sequence=torch.where(mask.unsqueeze(-1), a_cvrp.sequence, b.sequence),
            )
        case CVRPGraphSolution() as a_cvrp_graph:
            assert isinstance(b, CVRPGraphSolution)
            m = mask.unsqueeze(-1)
            return CVRPGraphSolution(
                _instance=a_cvrp_graph.instance,
                _next=torch.where(m, a_cvrp_graph.next_nodes, b.next_nodes),
                _prev=torch.where(m, a_cvrp_graph.prev_nodes, b.prev_nodes),
                _route_id=torch.where(m, a_cvrp_graph.route_ids, b.route_ids),
                batch_size=a_cvrp_graph.batch_size,
            )
        case TSPGraphSolution() as a_tsp:
            assert isinstance(b, TSPGraphSolution)
            m = mask.unsqueeze(-1)
            return TSPGraphSolution(
                _instance=a_tsp.instance,
                _next=torch.where(m, a_tsp.next_nodes, b.next_nodes),
                _prev=torch.where(m, a_tsp.prev_nodes, b.prev_nodes),
                batch_size=a_tsp.batch_size,
            )
        case _:
            raise NotImplementedError(f"select_solution not implemented for {type(a)}")
