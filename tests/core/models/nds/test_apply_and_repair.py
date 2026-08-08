import numpy as np
import pytest
import torch

from core.lns.numba.cvrp_apply_and_repair import apply_and_repair_parallel
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.insert.cvrp import CVRPGreedyInsertRepair
from core.models.nds.apply_and_repair import NumbaApplyAndRepair, TorchApplyAndRepair
from core.models.rollout import Rollout
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.solve import one_customer_per_route_graph
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.solve import random_permutation_graph

NUM_ROLLOUTS = 8
DECONSTRUCTION_SIZE = 10


def _rollout(solution, *, stitch: bool, seed: int) -> Rollout:
    """Wrap RandomNodeDestroy node sets into a Rollout (dummy log_p/entropy)."""
    destroy = RandomNodeDestroy(destruction_size=DECONSTRUCTION_SIZE, seed=seed, stitch=stitch)
    actions = destroy.select_nodes(solution, NUM_ROLLOUTS)  # (B, R, M)
    dummy = torch.zeros(actions.shape[:-1])  # (B, R)
    return Rollout.create(actions=actions, log_prob=dummy, entropy=dummy, entropy_norm=dummy)


@pytest.fixture
def cvrp_solution() -> CVRPGraphSolution:
    instance = CVRPSampler(CVRP.for_n(20), seed=42)(4)
    return one_customer_per_route_graph(instance)


@pytest.fixture
def tsp_solution() -> TSPGraphSolution:
    instance = TSPSampler(TSP(num_locations=20), seed=42)(4)
    return random_permutation_graph(instance)


class TestNumbaApplyAndRepairCVRP:
    def test_output_shape_complete_and_feasible(self, cvrp_solution: CVRPGraphSolution) -> None:
        rollout = _rollout(cvrp_solution, stitch=True, seed=1)
        result = NumbaApplyAndRepair(n_repairs=3, seed=7)(cvrp_solution, rollout)

        assert result.batch_size == torch.Size([4, NUM_ROLLOUTS])
        assert result.is_complete.all()
        assert result.is_feasible.all()
        assert (result.node_degree() == 2).all()
        assert result.objective().isfinite().all()
        assert (result.objective() > 0).all()

    def test_deterministic_for_fixed_seed(self, cvrp_solution: CVRPGraphSolution) -> None:
        rollout = _rollout(cvrp_solution, stitch=True, seed=1)
        first = NumbaApplyAndRepair(n_repairs=3, seed=7)(cvrp_solution, rollout)
        second = NumbaApplyAndRepair(n_repairs=3, seed=7)(cvrp_solution, rollout)

        assert torch.equal(first.next_nodes, second.next_nodes)

    def test_rebuilt_objective_matches_kernel(self, cvrp_solution: CVRPGraphSolution) -> None:
        """The rebuilt torch solution's objective equals the kernel's own objective."""
        rollout = _rollout(cvrp_solution, stitch=True, seed=1)
        result = NumbaApplyAndRepair(n_repairs=3, seed=7)(cvrp_solution, rollout)

        next_np, prev_np, locations, demand, capacity = cvrp_solution.to_numpy_arrays()
        nodes = rollout.actions.cpu().numpy().astype(np.int64)
        _, _, objective = apply_and_repair_parallel(
            next=next_np,
            prev=prev_np,
            nodes=nodes,
            locations=locations,
            demand=demand,
            capacity=capacity,
            n_repairs=3,
            base_seed=7,
        )

        assert torch.allclose(result.objective().double(), torch.from_numpy(objective), atol=1e-6)

    def test_quality_matches_torch(self, cvrp_solution: CVRPGraphSolution) -> None:
        """Same repair policy, independent RNG -> objectives close in aggregate."""
        rollout = _rollout(cvrp_solution, stitch=True, seed=1)
        torch_result = TorchApplyAndRepair(
            repair=CVRPGreedyInsertRepair(), n_repairs=5, stitch=True
        )(cvrp_solution, rollout)
        numba_result = NumbaApplyAndRepair(n_repairs=5, seed=7)(cvrp_solution, rollout)

        torch_mean = torch_result.objective().mean().item()
        numba_mean = numba_result.objective().mean().item()
        assert abs(torch_mean - numba_mean) / torch_mean < 0.05


class TestNumbaApplyAndRepairTSP:
    def test_output_shape_complete(self, tsp_solution: TSPGraphSolution) -> None:
        rollout = _rollout(tsp_solution, stitch=False, seed=1)
        result = NumbaApplyAndRepair(n_repairs=3, seed=7)(tsp_solution, rollout)

        assert result.batch_size == torch.Size([4, NUM_ROLLOUTS])
        assert result.is_complete.all()
        assert (result.node_degree() == 2).all()
        assert result.objective().isfinite().all()
        assert (result.objective() > 0).all()

    def test_deterministic_for_fixed_seed(self, tsp_solution: TSPGraphSolution) -> None:
        rollout = _rollout(tsp_solution, stitch=False, seed=1)
        first = NumbaApplyAndRepair(n_repairs=3, seed=7)(tsp_solution, rollout)
        second = NumbaApplyAndRepair(n_repairs=3, seed=7)(tsp_solution, rollout)

        assert torch.equal(first.next_nodes, second.next_nodes)

    def test_insert_in_affected_routes_only_rejected(self, tsp_solution: TSPGraphSolution) -> None:
        rollout = _rollout(tsp_solution, stitch=False, seed=1)
        op = NumbaApplyAndRepair(n_repairs=3, insert_in_affected_routes_only=True, seed=7)

        with pytest.raises(NotImplementedError):
            op(tsp_solution, rollout)
