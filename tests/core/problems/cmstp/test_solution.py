import torch

from core.problems.cmstp import CMSTPInstance, CMSTPSolution
from core.problems.cmstp.config import CMSTP
from core.problems.cmstp.sampler import CMSTPSampler
from core.states.destroy import initial_action_mask


def test_compute_subtree_ids_matches_branches():
    # depot(0) -> 1 -> {2, 3};  depot(0) -> 4 -> 5
    parent = torch.tensor([[-1, 0, 1, 1, 0, 4]])
    subtree = CMSTPSolution._compute_subtree_ids(parent, num_customers=5)
    assert subtree.tolist() == [[-1, 1, 1, 1, 4, 4]]


def test_star_start_is_complete_and_feasible():
    instance = CMSTPSampler(CMSTP(num_customers=20, capacity_raw=30), seed=0)(4)
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


def test_numpy_roundtrip_reproduces_subtree_ids():
    instance = CMSTPSampler(CMSTP(num_customers=20, capacity_raw=30), seed=0)(4)
    solution = CMSTPSolution.star(instance)
    arrays = solution.to_numpy()
    rebuilt = CMSTPSolution.from_numpy(
        instance,
        arrays,
        batch_size=instance.batch_size,
        device=instance.locations.device,
    )
    assert bool((rebuilt.subtree_ids == solution.subtree_ids).all())


def test_initial_action_mask_forbids_only_root():
    mask = initial_action_mask("cmstp", batch_size=3, num_nodes=6, device=torch.device("cpu"))
    assert bool(mask[:, 0].all())
    assert bool((~mask[:, 1:]).all())
