import pytest
import torch

from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.instance import CVRPInstance


def test_if_can_visualize_without_batch_dim():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)[0]
    assert instance.batch_dims == 0

    instance.visualize()


def test_if_can_visualize_with_batch_of_one():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)[:1]
    assert instance.batch_dims == 1

    instance.visualize()


def test_if_fails_to_visualize_batch():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)

    with pytest.raises(ValueError):
        instance.visualize()


def test_sampler_instance_has_unit_coordinate_scale():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(4)
    assert instance.coordinate_scale.shape == (4, 1)
    assert torch.all(instance.coordinate_scale == 1.0)
    assert instance.is_normalized


def test_normalize_scales_coordinates_and_records_scale():
    locations = torch.tensor([[[0.0, 0.0], [200.0, 0.0], [0.0, 400.0]]])  # (1, 3, 2)
    demand = torch.tensor([[0.0, 0.5, 0.5]])
    capacity = torch.ones(1, 1)
    instance = CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)

    assert instance.is_normalized  # scale starts at 1.0
    expected = locations.clone() / 400.0  # save before in-place modification
    instance.normalize_()

    assert torch.allclose(instance.coordinate_scale, torch.tensor([[400.0]]))
    assert torch.allclose(instance.locations, expected)
    assert not instance.is_normalized


def test_normalize_unbatched():
    locations = torch.tensor([[0.0, 0.0], [500.0, 0.0], [0.0, 300.0]])  # (3, 2)
    demand = torch.tensor([0.0, 0.5, 0.5])
    capacity = torch.ones(1)
    instance = CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)

    instance.normalize_()

    assert instance.coordinate_scale.shape == (1,)
    assert torch.allclose(instance.coordinate_scale, torch.tensor([500.0]))
    assert instance.locations.max() <= 1.0


def test_normalize_accumulates_scale_on_repeated_calls():
    locations = torch.tensor([[[0.0, 0.0], [100.0, 0.0]]])
    demand = torch.zeros(1, 2)
    capacity = torch.ones(1, 1)
    instance = CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)

    instance.normalize_()  # scale = 100, locations become [[0,0],[1,0]]
    instance.normalize_()  # scale = 1, locations unchanged

    assert torch.allclose(instance.coordinate_scale, torch.tensor([[100.0]]))
