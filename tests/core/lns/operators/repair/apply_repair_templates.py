from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest
import torch

from core.lns.acceptance_criterion import (
    GreedyAcceptance,
    SimulatedAnnealingAcceptance,
    UnconditionalAcceptance,
)
from core.lns.operators.destroy.random_node import RandomNodeDestroy

if TYPE_CHECKING:
    from core.lns.apply_and_repair import ApplyAndRepair
    from core.lns.operators.destroy.selection import NodeSelection
    from core.problems.solution import BaseSolution


class ApplyRepairContract:
    """Shared behaviour every numba apply-and-repair must satisfy."""

    num_rollouts: ClassVar[int] = 4
    n_repairs: ClassVar[int] = 3
    deconstruction_size: ClassVar[int] = 10
    destroy_seed: ClassVar[int] = 42
    repair_seed: ClassVar[int] = 7
    supports_affected_routes_only: ClassVar[bool] = False

    @pytest.fixture(params=[1, 4])
    def batch_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def complete_solution(self, batch_size: int) -> BaseSolution:
        raise NotImplementedError("Subclass must provide a `complete_solution` fixture.")

    def make_apply_repair(self, **kwargs: object) -> ApplyAndRepair:
        raise NotImplementedError("Subclass must implement `make_apply_repair`.")

    def make_selection(self, solution: BaseSolution, num_rollouts: int) -> NodeSelection:
        destroy = RandomNodeDestroy(
            destruction_size=self.deconstruction_size, seed=self.destroy_seed
        )
        return destroy.select(solution, num_rollouts)

    def solution_state(self, solution: BaseSolution) -> torch.Tensor:
        """A tensor whose exact values pin the solution structure, for determinism checks."""
        return solution.objective()

    # -- parallel ---------------------------------------------------------------

    def test_parallel_shape_and_complete(self, complete_solution: BaseSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        candidates = self.make_apply_repair().parallel(complete_solution, selection)

        expected = torch.Size([*complete_solution.batch_size, self.num_rollouts])
        assert candidates.batch_size == expected
        assert candidates.is_complete.all()

    def test_parallel_objective_is_finite_and_positive(
        self, complete_solution: BaseSolution
    ) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        objective = self.make_apply_repair().parallel(complete_solution, selection).objective()

        assert objective.isfinite().all()
        assert (objective > 0).all()

    def test_parallel_deterministic_for_fixed_seed(self, complete_solution: BaseSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        first = self.make_apply_repair().parallel(complete_solution, selection)
        second = self.make_apply_repair().parallel(complete_solution, selection)

        assert torch.equal(self.solution_state(first), self.solution_state(second))

    # -- sequential -------------------------------------------------------------

    def test_sequential_greedy_is_monotonic_and_complete(
        self, complete_solution: BaseSolution
    ) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        result = self.make_apply_repair().sequential(
            complete_solution, selection, GreedyAcceptance()
        )

        assert result.batch_size == complete_solution.batch_size
        assert result.is_complete.all()
        assert (result.objective() <= complete_solution.objective() + 1e-4).all()

    def test_sequential_deterministic_for_fixed_seed(self, complete_solution: BaseSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        first = self.make_apply_repair().sequential(
            complete_solution, selection, GreedyAcceptance()
        )
        second = self.make_apply_repair().sequential(
            complete_solution, selection, GreedyAcceptance()
        )

        assert torch.equal(self.solution_state(first), self.solution_state(second))

    def test_sequential_simulated_annealing_is_complete(
        self, complete_solution: BaseSolution
    ) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        criterion = SimulatedAnnealingAcceptance(temperature=0.1, decay=0.9)
        result = self.make_apply_repair().sequential(complete_solution, selection, criterion)

        assert result.is_complete.all()

    def test_sequential_unconditional_is_complete(self, complete_solution: BaseSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        result = self.make_apply_repair().sequential(
            complete_solution, selection, UnconditionalAcceptance()
        )

        assert result.is_complete.all()

    # -- affected-routes-only flag ---------------------------------------------

    def test_affected_routes_only_flag(self) -> None:
        if self.supports_affected_routes_only:
            pytest.skip(
                "Operator supports insert_in_affected_routes_only; see problem-specific test."
            )
        with pytest.raises(NotImplementedError):
            self.make_apply_repair(insert_in_affected_routes_only=True)


class FeasibleApplyRepairContract(ApplyRepairContract):
    def test_parallel_feasible(self, complete_solution: BaseSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        candidates = self.make_apply_repair().parallel(complete_solution, selection)

        assert candidates.is_feasible.all()

    def test_sequential_feasible(self, complete_solution: BaseSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        result = self.make_apply_repair().sequential(
            complete_solution, selection, GreedyAcceptance()
        )

        assert result.is_feasible.all()


class DegreeTwoApplyRepairMixin:
    """Every node has degree two -- the tour invariant of routing problems (CVRP/TSP)."""

    def test_parallel_every_node_has_degree_two(self, complete_solution: BaseSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        candidates = self.make_apply_repair().parallel(complete_solution, selection)

        assert (candidates.node_degree() == 2).all()

    def test_sequential_every_node_has_degree_two(self, complete_solution: BaseSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        result = self.make_apply_repair().sequential(
            complete_solution, selection, GreedyAcceptance()
        )

        assert (result.node_degree() == 2).all()
