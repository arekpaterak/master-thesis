import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance, UnconditionalAcceptance
from core.lns.improvement_step import ImprovementStep
from core.lns.operators.destroy.base import DestroyOperator
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.destroy.selection import EdgeSelection
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.lns.apply_repair import CVRPGreedyInsertApplyRepair
from core.problems.cvrp.solve import one_customer_per_route_graph

NUM_ROLLOUTS = 4


@pytest.fixture
def solution() -> CVRPGraphSolution:
    instance = CVRPSampler(CVRP.for_n(20), seed=42)(4)
    return one_customer_per_route_graph(instance)


def _destroy() -> RandomNodeDestroy:
    return RandomNodeDestroy[CVRPGraphSolution](destruction_size=10, seed=42)


def _apply_and_repair() -> CVRPGreedyInsertApplyRepair:
    return CVRPGreedyInsertApplyRepair(n_repairs=3, seed=7)


def _step(mode: str) -> ImprovementStep:
    return ImprovementStep(
        destroy=_destroy(),
        apply_and_repair=_apply_and_repair(),
        mode=mode,
        num_rollouts=NUM_ROLLOUTS,
    )


class TestParallelMode:
    def test_selects_argmin_over_k(self, solution: CVRPGraphSolution) -> None:
        """Unconditional acceptance must yield exactly the best-of-K parallel candidate."""
        result = _step("parallel")(solution, UnconditionalAcceptance())

        # Reconstruct the expected best-of-K independently (destroy is seeded, so the
        # selection is identical).
        selection = _destroy().select(solution, NUM_ROLLOUTS)
        candidates = _apply_and_repair().parallel(solution, selection)  # (*B, K)
        best = candidates.gather(-1, candidates.objective().argmin(dim=-1, keepdim=True)).squeeze(
            -1
        )

        torch.testing.assert_close(result.objective(), best.objective())

    def test_greedy_accept_is_monotonic(self, solution: CVRPGraphSolution) -> None:
        result = _step("parallel")(solution, GreedyAcceptance())

        assert result.batch_size == solution.batch_size
        assert (result.objective() <= solution.objective() + 1e-4).all()


class TestSequentialMode:
    def test_delegates_to_apply_and_repair_sequential(self, solution: CVRPGraphSolution) -> None:
        result = _step("sequential")(solution, GreedyAcceptance())

        # The step must call apply_and_repair.sequential with the same seeded selection.
        selection = _destroy().select(solution, NUM_ROLLOUTS)
        expected = _apply_and_repair().sequential(solution, selection, GreedyAcceptance())

        assert torch.equal(result.next_nodes, expected.next_nodes)


class _EdgeDestroy(DestroyOperator):
    """Stub destroy that emits an incompatible selection type."""

    selection_type = EdgeSelection

    def select(self, solution, num_rollouts):  # noqa: ANN001, ANN201
        raise NotImplementedError


def test_selection_type_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        ImprovementStep(
            destroy=_EdgeDestroy(),
            apply_and_repair=_apply_and_repair(),
            mode="parallel",
            num_rollouts=NUM_ROLLOUTS,
        )
