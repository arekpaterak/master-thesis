from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from core.lns.operators.repair.base import RepairOperator
    from core.problems.solution import BaseSolution


class RepairContract:
    """Shared behaviour every torch repair operator must satisfy."""

    @pytest.fixture
    def complete_solution(self) -> BaseSolution:
        raise NotImplementedError

    @pytest.fixture
    def deconstructed(self) -> BaseSolution:
        raise NotImplementedError

    def make_repair(self, *, seed: int | None = None) -> RepairOperator:
        raise NotImplementedError

    def test_repairs_to_complete(self, deconstructed: BaseSolution) -> None:
        assert not deconstructed.is_complete.any(), "Fixture must start incomplete."

        repaired = self.make_repair()(deconstructed)

        assert repaired.is_complete.all(), "Repaired solution must be complete."

    def test_objective_is_finite_and_positive(self, deconstructed: BaseSolution) -> None:
        objective = self.make_repair()(deconstructed).objective()

        assert objective.isfinite().all(), "Objective must be finite for a valid solution."
        assert (objective > 0).all(), "Objective must be positive."

    def test_already_complete_is_unchanged(self, complete_solution: BaseSolution) -> None:
        repaired = self.make_repair()(complete_solution)

        assert repaired.is_complete.all()
        assert torch.allclose(repaired.objective(), complete_solution.objective())


class FeasibleRepairContract(RepairContract):
    # TODO: But TSP also should be feasible after repair
    def test_repairs_to_feasible(self, deconstructed: BaseSolution) -> None:
        repaired = self.make_repair()(deconstructed)

        assert repaired.is_complete.all(), "Repaired solution must be complete."
        assert repaired.is_feasible.all(), "Repaired solution must be feasible."
