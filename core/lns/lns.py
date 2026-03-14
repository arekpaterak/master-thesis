from typing import NamedTuple

import torch
from tqdm import tqdm

from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.destroy import DestroyOperator
from core.lns.repair import RepairOperator
from core.problems.tsp.solution import TSPBatchSolution


class LNSResult(NamedTuple):
    best_solutions: TSPBatchSolution
    initial_lengths: torch.Tensor  # (batch,)
    best_lengths: torch.Tensor  # (batch,)
    history: torch.Tensor  # (n_iterations+1, batch)
    improvements: torch.Tensor


@torch.no_grad()
def lns(
    starting_solutions: TSPBatchSolution,
    destroy: DestroyOperator,
    repair: RepairOperator,
    accept: AcceptanceCriterion,
    n_iterations: int,
    *,
    with_progress: bool = False,
) -> LNSResult:
    current = starting_solutions
    best = starting_solutions
    history = [starting_solutions.tour_lengths.cpu()]

    iter_bar = tqdm(range(n_iterations), desc="LNS", disable=not with_progress)
    for _ in iter_bar:
        partial = destroy(current)
        candidate = repair(partial)
        mask = accept(current, candidate)  # (batch,) bool

        # update current and best using tensor masking
        current = _select(mask, candidate, current)
        best = _select(current.tour_lengths < best.tour_lengths, current, best)
        history.append(best.tour_lengths.detach().cpu())

    return LNSResult(
        best_solutions=best,
        initial_lengths=starting_solutions.tour_lengths.cpu(),
        best_lengths=best.tour_lengths.cpu(),
        history=torch.stack(history, dim=0),
        improvements=(1 - best.tour_lengths.cpu() / starting_solutions.tour_lengths.cpu()),
    )


def _select(mask: torch.Tensor, a: TSPBatchSolution, b: TSPBatchSolution) -> TSPBatchSolution:
    """Select a[i] where mask[i] is True, else b[i]."""
    return TSPBatchSolution(
        instance=a.instance,  # shared, should not change
        tour=torch.where(mask.unsqueeze(1), a.tour, b.tour),
    )
