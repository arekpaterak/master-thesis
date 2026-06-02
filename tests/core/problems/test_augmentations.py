import pytest
import torch

from core.problems.augmentations import NUM_D4_AUGMENTATIONS, apply_d4_transforms
from core.problems.cvrp.instance import CVRPInstance
from core.problems.tsp.instance import TSPInstance


@pytest.mark.parametrize("n_augmentations", [1, 2, 4, 8])
def test_d4_transforms_preserve_pairwise_distances(n_augmentations):
    torch.manual_seed(0)
    B, N = 3, 12
    locations = torch.rand(B, N, 2)

    transformed = apply_d4_transforms(locations, n_augmentations)
    assert transformed.shape == (B, n_augmentations, N, 2)

    original_distances = torch.cdist(locations, locations)
    augmented_distances = torch.cdist(transformed, transformed)
    diff = (augmented_distances - original_distances.unsqueeze(-3)).abs().max().item()
    assert diff < 1e-5


def test_d4_transforms_preserves_identity_at_index_zero():
    locations = torch.rand(2, 5, 2)
    out = apply_d4_transforms(locations, 1)
    assert torch.allclose(out, locations.unsqueeze(-3))


def test_d4_transforms_rejects_out_of_range():
    locations = torch.rand(1, 3, 2)
    with pytest.raises(AssertionError):
        apply_d4_transforms(locations, 0)
    with pytest.raises(AssertionError):
        apply_d4_transforms(locations, NUM_D4_AUGMENTATIONS + 1)


def test_cvrp_instance_augment_preserves_distances_and_broadcasts_other_fields():
    torch.manual_seed(0)
    B, N = 4, 10
    locations = torch.rand(B, N + 1, 2)
    demand = torch.cat([torch.zeros(B, 1), torch.rand(B, N) * 5], dim=-1)
    capacity = torch.ones(B, 1) * 30.0

    instance = CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)
    augmented = instance.augment(8)

    assert augmented.batch_size == torch.Size([B, 8])
    assert augmented.locations.shape == (B, 8, N + 1, 2)
    diff = (augmented.distances() - instance.distances().unsqueeze(-3)).abs().max().item()
    assert diff < 1e-5
    assert torch.equal(augmented.demand, demand.unsqueeze(-2).expand(B, 8, N + 1))
    assert torch.equal(augmented.capacity, capacity.unsqueeze(-2).expand(B, 8, 1))


def test_tsp_instance_augment_preserves_distances():
    torch.manual_seed(0)
    B, N = 4, 10
    locations = torch.rand(B, N, 2)

    instance = TSPInstance.create(locations=locations)
    augmented = instance.augment(4)

    assert augmented.batch_size == torch.Size([B, 4])
    diff = (augmented.distances() - instance.distances().unsqueeze(-3)).abs().max().item()
    assert diff < 1e-5
