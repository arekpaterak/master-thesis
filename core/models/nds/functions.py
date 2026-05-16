from __future__ import annotations

from collections.abc import Callable

import torch

from core.lns.operators.repair.greedy import CVRPSequentialGreedyNodeInsertionRepair
from core.lns.operators.repair.greedy_edge import TSPGreedyEdgeRepair
from core.models.nds.model import NDSModel
from core.models.rollout import Rollout
from core.problems.cvrp import CVRPSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.graph_solution import TSPGraphSolution

ImprovementStep = Callable[[BaseSolution, NDSModel], BaseSolution]

ApplyRollout = Callable[[BaseSolution, Rollout], BaseSolution]

Repair = Callable[[BaseSolution, Rollout], BaseSolution]


def noop_improvement_step(
    solution: BaseSolution,
    model: NDSModel,
) -> BaseSolution:
    return solution


def make_improvement_step(
    *,
    apply_rollout_fn: ApplyRollout,
    repair: Repair,
    num_rollouts: int,
    deconstruction_size: int,
) -> ImprovementStep:
    """
    Build an improvement step following the NDS procedure (Hottung et al. 2025, Figure 2).

    Generates num_rollouts deconstruction rollouts from the current solution in a single
    batched model call, then applies them sequentially with greedy acceptance: each rollout
    deconstructs and repairs the evolving current solution, retaining the result only when
    it improves the objective.
    """

    @torch.no_grad()
    def improvement_step(solution: BaseSolution, model: NDSModel) -> BaseSolution:
        model.eval()
        rollout = model.rollout(
            solution,
            rollouts_per_solution=num_rollouts,
            deconstruction_size=deconstruction_size,
            greedy=False,
        )
        # rollout.actions: (B, num_rollouts, deconstruction_size)
        current = solution
        for r in range(num_rollouts):
            single_rollout = Rollout.create(
                actions=rollout.actions[:, r, :],
                log_p=rollout.log_p[:, r],
                entropy=rollout.entropy[:, r],
                entropy_norm=rollout.entropy_norm[:, r],
            )
            partial = apply_rollout_fn(current, single_rollout)
            candidate = repair(partial, single_rollout)
            accept = candidate.objective() < current.objective()
            current = select_solution(accept, candidate, current)
        return current

    return improvement_step


def select_solution(mask: torch.Tensor, a: BaseSolution, b: BaseSolution) -> BaseSolution:
    """Select a[i] where mask[i] is True, else b[i]."""
    match a:
        case CVRPSolution() as a_cvrp:
            assert isinstance(b, CVRPSolution)
            return CVRPSolution.create(
                instance=a_cvrp.instance,
                sequence=torch.where(mask.unsqueeze(-1), a_cvrp.sequence, b.sequence),
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


def apply_rollout(solution: BaseSolution, rollout: Rollout) -> BaseSolution:
    match solution:
        case CVRPSolution() as s:
            return s.remove_customers(rollout.actions)
        case TSPGraphSolution() as s:
            return s.remove_nodes(rollout.actions)
        case _:
            raise NotImplementedError(f"Not implemented for {type(solution)}")


def greedy_cvrp_repair(solution: CVRPSolution, rollout: Rollout) -> CVRPSolution:
    return CVRPSequentialGreedyNodeInsertionRepair()(solution)


def greedy_tsp_repair(solution: TSPGraphSolution, rollout: Rollout) -> TSPGraphSolution:
    return TSPGreedyEdgeRepair()(solution)


def only_expanding_apply_rollout(solution: BaseSolution, rollout: Rollout) -> BaseSolution:
    """
    Only expands the ``solution`` from (B, destruction_size) to (B, num_rollouts, destruction_size), based on the ``rollout`` dimensionality.
    """
    return solution.unsqueeze(1).expand(*solution.batch_size, rollout.batch_size[1])


def noop_repair(
    solution: BaseSolution,
    rollout: Rollout,
) -> BaseSolution:
    return solution
