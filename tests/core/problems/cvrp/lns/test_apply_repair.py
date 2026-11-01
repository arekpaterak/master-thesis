import numpy as np
import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance
from core.lns.apply_and_repair import TorchApplyAndRepair
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.lns._numba import apply_and_repair_parallel, apply_and_repair_sequential
from core.problems.cvrp.lns.apply_repair import CVRPGreedyInsertApplyRepair
from core.problems.cvrp.lns.arrays import CVRPInstanceArrays, CVRPSolutionArrays
from core.problems.cvrp.lns.operators.repair.insert import CVRPGreedyInsertRepair
from core.problems.cvrp.solve import one_customer_per_route_graph
from tests.core.lns.operators.repair.apply_repair_templates import (
    DegreeTwoApplyRepairMixin,
    FeasibleApplyRepairContract,
)


class TestCVRPApplyRepair(FeasibleApplyRepairContract, DegreeTwoApplyRepairMixin):
    supports_affected_routes_only = True

    @pytest.fixture
    def complete_solution(self, batch_size: int) -> CVRPGraphSolution:
        instance = CVRPSampler(CVRPDistribution.for_n(20), seed=42)(batch_size)
        return one_customer_per_route_graph(instance)

    def make_apply_repair(self, **kwargs: object) -> CVRPGreedyInsertApplyRepair:
        kwargs.setdefault("n_repairs", self.n_repairs)
        kwargs.setdefault("seed", self.repair_seed)
        return CVRPGreedyInsertApplyRepair(**kwargs)

    def solution_state(self, solution: CVRPGraphSolution) -> torch.Tensor:
        return solution.next_nodes

    # -- CVRP specifics ---------------------------------------------------------

    def test_sequential_depot_invariant_preserved(
        self, complete_solution: CVRPGraphSolution
    ) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        repaired = self.make_apply_repair().sequential(
            complete_solution, selection, GreedyAcceptance()
        )

        assert (repaired.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (repaired.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()

    def test_with_multidim_batch(self) -> None:
        instance = CVRPSampler(CVRPDistribution.for_n(20), seed=1)(3).augment(2)
        solution = one_customer_per_route_graph(instance)
        assert solution.batch_size == torch.Size([3, 2])

        selection = self.make_selection(solution, self.num_rollouts)
        repaired = self.make_apply_repair().sequential(solution, selection, GreedyAcceptance())

        assert repaired.batch_size == torch.Size([3, 2])
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_restricted_to_affected_routes_is_complete_and_feasible(
        self, complete_solution: CVRPGraphSolution
    ) -> None:
        selection = self.make_selection(complete_solution, self.num_rollouts)
        repaired = self.make_apply_repair(insert_in_affected_routes_only=True).sequential(
            complete_solution, selection, GreedyAcceptance()
        )

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()
        assert (repaired.node_degree() == 2).all()

    # -- torch-vs-numba parity --------------------------------------------------

    def test_repaired_objective_matches_numba(self) -> None:
        """The rebuilt torch solution's objective equals the numba's objective."""
        solution = one_customer_per_route_graph(CVRPSampler(CVRPDistribution.for_n(20), seed=42)(4))
        selection = RandomNodeDestroy(destruction_size=10, seed=1).select(solution, 8)
        result = CVRPGreedyInsertApplyRepair(n_repairs=3, seed=7).parallel(solution, selection)

        nodes = selection.nodes.cpu().numpy().astype(np.int64)
        first_base_seed = int(np.random.default_rng(7).integers(2**31))
        _out, objective = apply_and_repair_parallel(
            solution.to_numpy(),
            solution.instance.to_numpy(),
            nodes,
            n_repairs=3,
            base_seed=first_base_seed,
        )

        assert torch.allclose(result.objective().double(), torch.from_numpy(objective), atol=1e-6)

    def test_each_call_draws_new_randomness_reproducibly(self) -> None:
        solution = one_customer_per_route_graph(CVRPSampler(CVRPDistribution.for_n(20), seed=42)(4))
        selection = RandomNodeDestroy(destruction_size=10, seed=1).select(solution, 8)
        a = CVRPGreedyInsertApplyRepair(n_repairs=1, seed=7)
        b = CVRPGreedyInsertApplyRepair(n_repairs=1, seed=7)

        first, second = a.parallel(solution, selection), a.parallel(solution, selection)

        assert not torch.equal(first.objective(), second.objective())
        assert torch.equal(first.objective(), b.parallel(solution, selection).objective())
        assert torch.equal(second.objective(), b.parallel(solution, selection).objective())

    def test_quality_matches_torch(self) -> None:
        """Same repair policy, independent RNG -> objectives close in aggregate."""
        solution = one_customer_per_route_graph(CVRPSampler(CVRPDistribution.for_n(20), seed=42)(4))
        selection = RandomNodeDestroy(destruction_size=10, seed=1).select(solution, 8)

        torch_result = TorchApplyAndRepair(repair=CVRPGreedyInsertRepair(), n_repairs=5).parallel(
            solution, selection
        )
        numba_result = CVRPGreedyInsertApplyRepair(n_repairs=5, seed=7).parallel(
            solution, selection
        )

        torch_mean = torch_result.objective().mean().item()
        numba_mean = numba_result.objective().mean().item()
        assert abs(torch_mean - numba_mean) / torch_mean < 0.05


def _two_route_problem() -> tuple[CVRPSolutionArrays, CVRPInstanceArrays, np.ndarray, dict]:
    """Single element, two routes A=[1,2], B=[3,4]. Customer 1 sits right next to
    customer 3 of the *other* route, so unrestricted repair pulls it into route B.
    """
    solution = CVRPSolutionArrays(
        next=np.array([[-1, 2, 0, 4, 0]], dtype=np.int64),
        prev=np.array([[-1, 0, 1, 0, 3]], dtype=np.int64),
    )
    instance = CVRPInstanceArrays(
        locations=np.array(
            [[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [5.0, 0.1], [5.0, 5.1]]], dtype=np.float64
        ),
        demand=np.array([[0.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        capacity=np.array([100.0], dtype=np.float64),  # never binds
    )
    nodes_to_remove = np.array([[[1]]], dtype=np.int64)  # remove customer 1 from route A
    params = dict(n_repairs=1, accept_kind=0, temperature=0.0, eps=1e-6, base_seed=0)
    return solution, instance, nodes_to_remove, params


class TestInsertInAffectedRoutesOnly:
    def test_unrestricted_pulls_customer_into_untouched_route(self) -> None:
        solution, instance, nodes, params = _two_route_problem()
        out = apply_and_repair_sequential(
            solution, instance, nodes, insert_in_affected_routes_only=False, **params
        )

        # Geometry check: without the restriction, customer 1 lands next to customer 3,
        # i.e. inside the route it did not come from.
        assert out.next[0, 1] == 3

    def test_restricted_keeps_customer_out_of_untouched_route(self) -> None:
        solution, instance, nodes, params = _two_route_problem()
        out = apply_and_repair_sequential(
            solution, instance, nodes, insert_in_affected_routes_only=True, **params
        )
        next_out, prev_out = out.next, out.prev

        # Untouched route B stays exactly intact.
        assert next_out[0, 3] == 4 and prev_out[0, 3] == 0
        assert next_out[0, 4] == 0 and prev_out[0, 4] == 3
        # Customer 1 is not adjacent to any route-B node (3 or 4).
        assert next_out[0, 1] not in (3, 4)
        assert prev_out[0, 1] not in (3, 4)
