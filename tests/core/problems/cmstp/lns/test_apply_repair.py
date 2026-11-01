import pytest
import torch

from core.lns.acceptance_criterion import UnconditionalAcceptance
from core.problems.cmstp import CMSTPSolution
from core.problems.cmstp.config import CMSTPDistribution
from core.problems.cmstp.construction import star
from core.problems.cmstp.lns.apply_repair import CMSTPGreedyInsertApplyRepair
from core.problems.cmstp.sampler import CMSTPSampler
from tests.core.lns.operators.repair.apply_repair_templates import FeasibleApplyRepairContract


class TestCMSTPApplyRepair(FeasibleApplyRepairContract):
    @pytest.fixture
    def complete_solution(self, batch_size: int) -> CMSTPSolution:
        instance = CMSTPSampler(CMSTPDistribution(num_customers=20, capacity_raw=30), seed=0)(
            batch_size
        )
        return star(instance)

    def make_apply_repair(self, **kwargs: object) -> CMSTPGreedyInsertApplyRepair:
        kwargs.setdefault("n_repairs", self.n_repairs)
        kwargs.setdefault("seed", self.repair_seed)
        return CMSTPGreedyInsertApplyRepair(**kwargs)

    def solution_state(self, solution: CMSTPSolution) -> torch.Tensor:
        return solution.parent_nodes

    # -- CMSTP specifics --------------------------------------------------------

    def test_parallel_never_worsens_star(self, complete_solution: CMSTPSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        candidates = self.make_apply_repair().parallel(complete_solution, selection)

        # Removing from the star and greedily reattaching can only match or improve it.
        assert (candidates.objective() <= complete_solution.objective().unsqueeze(-1) + 1e-6).all()

    def test_sequential_unconditional_is_feasible(self, complete_solution: CMSTPSolution) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        result = self.make_apply_repair().sequential(
            complete_solution, selection, UnconditionalAcceptance()
        )

        assert result.is_feasible.all()
