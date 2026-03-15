import torch

from core.problems.cvrp.config import CVRPConfig
from core.problems.cvrp.sampler import CVRPBatchSampler


def test_sampler_shape():
    config = CVRPConfig.for_n(20)
    sampler = CVRPBatchSampler(config)
    batch_size = 5

    sample = sampler(batch_size)

    assert sample.batch_size[0] == batch_size, (
        "Expected sample batch to have the same batch size as requested"
    )
    assert sample.locations.shape == (batch_size, config.num_customers + 1, 2), (
        "Expected sample locations to have shape (batch_size, num_customers + 1, 2)"
    )
    assert sample.demand.shape == (batch_size, config.num_customers + 1), (
        "Expected sample demand to have shape (batch_size, num_customers + 1)"
    )
    assert sample.capacity.shape == (batch_size, 1), (
        "Expected sample capacity to have shape (batch_size, 1)"
    )


def test_sampler_value_range():
    config = CVRPConfig.for_n(20)
    sampler = CVRPBatchSampler(config)
    batch_size = 5

    sample = sampler(batch_size)

    assert torch.all(sample.locations >= config.min_coordinate) and torch.all(
        sample.locations <= config.max_coordinate
    ), "Expected all sample locations to be within the specified coordinate range"

    assert torch.all(sample.demand[:, 0] == 0), "Expected depot demand to be exactly 0"
    assert torch.all(
        sample.demand[:, 1:] >= (config.min_demand_raw / config.vehicle_capacity_raw)
    ) and torch.all(
        sample.demand[:, 1:] <= (config.max_demand_raw / config.vehicle_capacity_raw)
    ), "Expected customer demands to be within the configured normalized demand range"

    assert torch.all(sample.capacity == 1.0), "Expected normalized capacities to be 1.0"


def test_sampler_determinism():
    config = CVRPConfig.for_n(20)
    seed = 42
    sampler1 = CVRPBatchSampler(config, seed=seed)
    sampler2 = CVRPBatchSampler(config, seed=seed)
    batch_size = 5

    sample1 = sampler1(batch_size)
    sample2 = sampler2(batch_size)

    assert torch.allclose(sample1.locations, sample2.locations), (
        "Expected samples from two samplers with the same seed to be identical"
    )
    assert torch.allclose(sample1.demand, sample2.demand), (
        "Expected demands from two samplers with the same seed to be identical"
    )
    assert torch.allclose(sample1.capacity, sample2.capacity), (
        "Expected capacities from two samplers with the same seed to be identical"
    )


def test_sampler_different_seeds():
    config = CVRPConfig.for_n(20)
    seed1 = 42
    seed2 = 43
    sampler1 = CVRPBatchSampler(config, seed=seed1)
    sampler2 = CVRPBatchSampler(config, seed=seed2)
    batch_size = 5

    sample1 = sampler1(batch_size)
    sample2 = sampler2(batch_size)

    assert not torch.allclose(sample1.locations, sample2.locations), (
        "Expected samples from two samplers with different seeds to be different"
    )
