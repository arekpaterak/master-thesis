import torch

from core.problems.cmstp import CMSTPInstance
from core.problems.cmstp.config import CMSTPDistribution
from core.problems.cmstp.construction import esau_williams, star
from core.problems.cmstp.sampler import CMSTPSampler


def _random_batch(num_customers: int = 20, batch_size: int = 8) -> CMSTPInstance:
    return CMSTPSampler(CMSTPDistribution(num_customers=num_customers, capacity_raw=30), seed=0)(
        batch_size
    )


def test_esau_williams_is_complete_and_feasible():
    instance = _random_batch()
    solution = esau_williams(instance)
    assert bool(solution.is_complete.all())
    assert bool(solution.is_feasible.all())


def test_esau_williams_never_worse_than_star():
    instance = _random_batch()
    ew = esau_williams(instance).objective()
    trivial = star(instance).objective()
    assert bool((ew <= trivial + 1e-6).all())


def test_esau_williams_is_deterministic():
    instance = _random_batch()
    first = esau_williams(instance)
    second = esau_williams(instance)
    assert bool((first.parent_nodes == second.parent_nodes).all())


def test_esau_williams_merges_obvious_pair():
    # Depot at origin; two customers far from the depot but close to each other.
    # Merging 2 under 1 saves ~its whole gate edge, so EW must chain them.
    instance = CMSTPInstance.create(
        locations=torch.tensor([[[0.0, 0.0], [10.0, 0.0], [11.0, 0.0]]]),
        demand=torch.tensor([[0.0, 1.0, 1.0]]),
        capacity=torch.tensor([[10.0]]),
    )
    solution = esau_williams(instance)
    assert solution.parent_nodes.tolist() == [[-1, 0, 1]]
    assert bool(solution.is_feasible.all())


def test_esau_williams_handles_unbatched_instance():
    batched = _random_batch(batch_size=1)
    unbatched = batched[0]
    solution = esau_williams(unbatched)
    assert solution.parent_nodes.shape == (unbatched.num_customers + 1,)
    assert bool(solution.is_feasible.all())
