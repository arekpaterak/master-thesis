from typing import NamedTuple

import torch
from tqdm import tqdm

from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.destroy import DestroyOperator
from core.lns.repair import RepairOperator
from core.problems.tsp.solution import TSPSolution


class LNSResult(NamedTuple):
    best_solutions: TSPSolution
    initial_lengths: torch.Tensor  # (batch,)
    best_lengths: torch.Tensor  # (batch,)
    history: torch.Tensor  # (n_iterations+1, batch)
    improvements: torch.Tensor


@torch.no_grad()
def lns(
    starting_solutions: TSPSolution,
    destroy: DestroyOperator,
    repair: RepairOperator,
    accept: AcceptanceCriterion,
    n_iterations: int,
    *,
    with_progress: bool = False,
) -> LNSResult:
    current = starting_solutions
    best = starting_solutions
    history = [starting_solutions.tour_length.cpu()]

    iter_bar = tqdm(range(n_iterations), desc="LNS", disable=not with_progress)
    for _ in iter_bar:
        partial = destroy(current)
        candidate = repair(partial)
        mask = accept(current, candidate)  # (batch,) bool

        # update current and best using tensor masking
        current = _select(mask, candidate, current)
        best = _select(current.tour_length < best.tour_length, current, best)
        history.append(best.tour_length.detach().cpu())

    return LNSResult(
        best_solutions=best,
        initial_lengths=starting_solutions.tour_length.cpu(),
        best_lengths=best.tour_length.cpu(),
        history=torch.stack(history, dim=0),
        improvements=(1 - best.tour_length.cpu() / starting_solutions.tour_length.cpu()),
    )


def _select(mask: torch.Tensor, a: TSPSolution, b: TSPSolution) -> TSPSolution:
    """Select a[i] where mask[i] is True, else b[i]."""
    return TSPSolution(
        instance=a.instance,  # shared, should not change
        tour=torch.where(mask.unsqueeze(1), a.tour, b.tour),
    )
