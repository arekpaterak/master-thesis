import numpy as np
import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance, SimulatedAnnealingAcceptance
from core.lns.improvement_step import NumbaSequentialImprovementStep
from core.lns.numba_kernels.cvrp_sequential_improvement_step import run_with_removals
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.insert.cvrp import CVRPGreedyInsertRepair
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.solve import one_customer_per_route_graph


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    return request.param


@pytest.fixture
def solution(batch_size: int) -> CVRPGraphSolution:
    instance = CVRPSampler(CVRP.for_n(20), seed=42)(batch_size)
    return one_customer_per_route_graph(instance)


def _make_improvement_step(
    num_rollouts: int = 4, n_repairs: int = 3
) -> NumbaSequentialImprovementStep:
    destroy = RandomNodeDestroy[CVRPGraphSolution](destruction_size=10, seed=42, stitch=True)
    return NumbaSequentialImprovementStep(
        destroy=destroy,
        repair=CVRPGreedyInsertRepair(),
        num_rollouts=num_rollouts,
        n_repairs=n_repairs,
        seed=7,
    )


class TestNumbaSequentialImprovementStep:
    def test_complete_and_feasible(self, solution: CVRPGraphSolution) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert repaired.batch_size == solution.batch_size
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_objective_is_finite_and_positive(self, solution: CVRPGraphSolution) -> None:
        objective = _make_improvement_step()(solution, GreedyAcceptance()).objective()

        assert objective.isfinite().all()
        assert (objective > 0).all()

    def test_objective_is_monotonic_with_greedy_acceptance(
        self, solution: CVRPGraphSolution
    ) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert (repaired.objective() <= solution.objective() + 1e-4).all()

    def test_depot_invariant_preserved(self, solution: CVRPGraphSolution) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert (repaired.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (repaired.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()

    def test_every_customer_has_degree_of_two(self, solution: CVRPGraphSolution) -> None:
        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert (repaired.node_degree() == 2).all()

    def test_with_simulated_annealing(self, solution: CVRPGraphSolution) -> None:
        criterion = SimulatedAnnealingAcceptance(temperature=0.1, decay=0.9)
        repaired = _make_improvement_step()(solution, criterion)

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_with_multidim_batch(self) -> None:
        instance = CVRPSampler(CVRP.for_n(20), seed=1)(3).augment(2)
        solution = one_customer_per_route_graph(instance)
        assert solution.batch_size == torch.Size([3, 2])

        repaired = _make_improvement_step()(solution, GreedyAcceptance())

        assert repaired.batch_size == torch.Size([3, 2])
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_if_deterministic_for_fixed_seed(self, solution: CVRPGraphSolution) -> None:
        first = _make_improvement_step()(solution, GreedyAcceptance())
        second = _make_improvement_step()(solution, GreedyAcceptance())

        assert torch.equal(first.next_nodes, second.next_nodes)

    def test_rejects_non_random_node_destroy(self) -> None:
        class _Dummy:
            pass

        with pytest.raises(NotImplementedError):
            NumbaSequentialImprovementStep(destroy=_Dummy(), repair=CVRPGreedyInsertRepair())


def _two_route_problem() -> dict:
    """Single element, two routes A=[1,2], B=[3,4]. Customer 1 sits right next to
    customer 3 of the *other* route, so unrestricted repair pulls it into route B.
    """
    next = np.array([[-1, 2, 0, 4, 0]], dtype=np.int64)
    prev = np.array([[-1, 0, 1, 0, 3]], dtype=np.int64)
    locations = np.array(
        [[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [5.0, 0.1], [5.0, 5.1]]], dtype=np.float64
    )
    demand = np.array([[0.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    capacity = np.array([100.0], dtype=np.float64)  # never binds
    nodes_to_remove = np.array([[[1]]], dtype=np.int64)  # remove customer 1 from route A
    return dict(
        next=next,
        prev=prev,
        nodes_to_remove=nodes_to_remove,
        locations=locations,
        demand=demand,
        capacity=capacity,
        n_repairs=1,
        accept_kind=0,  # greedy
        temperature=0.0,
        eps=1e-6,
        base_seed=0,
    )


class TestInsertInNewRoutesOnly:
    def test_unrestricted_pulls_customer_into_untouched_route(self) -> None:
        next_out, _ = run_with_removals(**_two_route_problem(), insert_in_new_routes_only=False)

        # Geometry check: without the restriction, customer 1 lands next to customer 3,
        # i.e. inside the route it did not come from.
        assert next_out[0, 1] == 3

    def test_restricted_keeps_customer_out_of_untouched_route(self) -> None:
        next_out, prev_out = run_with_removals(
            **_two_route_problem(), insert_in_new_routes_only=True
        )

        # Untouched route B stays exactly intact.
        assert next_out[0, 3] == 4 and prev_out[0, 3] == 0
        assert next_out[0, 4] == 0 and prev_out[0, 4] == 3
        # Customer 1 is not adjacent to any route-B node (3 or 4).
        assert next_out[0, 1] not in (3, 4)
        assert prev_out[0, 1] not in (3, 4)

    def test_restricted_repair_is_complete_and_feasible(self) -> None:
        instance = CVRPSampler(CVRP.for_n(20), seed=11)(4)
        solution = one_customer_per_route_graph(instance)
        destroy = RandomNodeDestroy[CVRPGraphSolution](destruction_size=10, seed=3, stitch=True)
        step = NumbaSequentialImprovementStep(
            destroy=destroy,
            repair=CVRPGreedyInsertRepair(),
            num_rollouts=4,
            n_repairs=3,
            seed=7,
            insert_in_new_routes_only=True,
        )

        repaired = step(solution, GreedyAcceptance())

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()
        assert (repaired.node_degree() == 2).all()
