import numpy as np
import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance, UnconditionalAcceptance
from core.lns.operators.destroy.selection import NodeSelection
from core.problems.cmstp import CMSTPInstance, CMSTPSolution
from core.problems.cmstp.config import CMSTP
from core.problems.cmstp.lns.apply_repair import CMSTPGreedyInsertApplyRepair
from core.problems.cmstp.sampler import CMSTPSampler
from core.states.destroy import initial_action_mask


@pytest.fixture
def instance():
    return CMSTPSampler(CMSTP(num_customers=20, capacity_raw=30), seed=0)(4)


def _random_nodes(num_instances, num_rollouts, deconstruction_size, num_customers, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            [
                rng.choice(np.arange(1, num_customers + 1), size=deconstruction_size, replace=False)
                for _ in range(num_rollouts)
            ]
            for _ in range(num_instances)
        ]
    ).astype(np.int64)


def test_compute_subtree_ids_matches_branches():
    # depot(0) -> 1 -> {2, 3};  depot(0) -> 4 -> 5
    parent = torch.tensor([[-1, 0, 1, 1, 0, 4]])
    subtree = CMSTPSolution._compute_subtree_ids(parent, num_customers=5)
    assert subtree.tolist() == [[-1, 1, 1, 1, 4, 4]]


def test_star_start_is_complete_and_feasible(instance):
    solution = CMSTPSolution.star(instance)
    assert bool(solution.is_complete.all())
    assert bool(solution.is_feasible.all())
    # Every customer is a direct child of the depot -> its own subtree.
    assert bool((solution.parent_nodes[..., 1:] == CMSTPSolution.DEPOT).all())


def test_node_degree_counts_variable_fan_out():
    inst = CMSTPInstance.create(
        locations=torch.rand(1, 6, 2),
        demand=torch.tensor([[0.0, 1, 1, 1, 1, 1]]),
        capacity=torch.tensor([[10.0]]),
    )
    parent = torch.tensor([[-1, 0, 1, 1, 0, 4]])
    subtree = CMSTPSolution._compute_subtree_ids(parent, num_customers=5)
    solution = CMSTPSolution(_instance=inst, _parent=parent, _subtree_id=subtree, batch_size=[1])
    # node 1 has parent (depot) + two children (2, 3) -> degree 3.
    assert solution.node_degree().tolist() == [[3, 1, 1, 2, 1]]


def test_numpy_roundtrip_reproduces_subtree_ids(instance):
    solution = CMSTPSolution.star(instance)
    arrays = solution.to_numpy()
    rebuilt = CMSTPSolution.from_numpy(
        instance,
        arrays,
        batch_size=instance.batch_size,
        device=instance.locations.device,
    )
    assert bool((rebuilt.subtree_ids == solution.subtree_ids).all())


def _selection(nodes: np.ndarray) -> NodeSelection:
    return NodeSelection(torch.from_numpy(nodes))


def test_apply_and_repair_parallel_is_feasible_and_improves(instance):
    solution = CMSTPSolution.star(instance)
    num_rollouts, deconstruction_size = 2, 5
    nodes = _random_nodes(
        instance.batch_size[0], num_rollouts, deconstruction_size, instance.num_customers
    )
    op = CMSTPGreedyInsertApplyRepair(n_repairs=3, seed=0)
    candidates = op.parallel(solution, _selection(nodes))

    assert tuple(candidates.batch_size) == (instance.batch_size[0], num_rollouts)
    assert bool(candidates.is_complete.all())
    assert bool(candidates.is_feasible.all())
    # Removing from the star and greedily reattaching can only match or improve it.
    assert bool((candidates.objective() <= solution.objective().unsqueeze(-1) + 1e-6).all())


def test_apply_and_repair_sequential_greedy_never_worsens(instance):
    solution = CMSTPSolution.star(instance)
    nodes = _random_nodes(instance.batch_size[0], 8, 5, instance.num_customers, seed=1)
    op = CMSTPGreedyInsertApplyRepair(n_repairs=3, seed=7)
    result = op.sequential(solution, _selection(nodes), GreedyAcceptance())
    assert bool(result.is_complete.all())
    assert bool(result.is_feasible.all())
    assert bool((result.objective() <= solution.objective() + 1e-6).all())


def test_apply_and_repair_sequential_unconditional_is_feasible(instance):
    solution = CMSTPSolution.star(instance)
    nodes = _random_nodes(instance.batch_size[0], 8, 5, instance.num_customers, seed=1)
    op = CMSTPGreedyInsertApplyRepair(n_repairs=3, seed=7)
    result = op.sequential(solution, _selection(nodes), UnconditionalAcceptance())
    assert bool(result.is_complete.all())
    assert bool(result.is_feasible.all())


def test_initial_action_mask_forbids_only_root():
    mask = initial_action_mask("cmstp", batch_size=3, num_nodes=6, device=torch.device("cpu"))
    assert bool(mask[:, 0].all())
    assert bool((~mask[:, 1:]).all())
