import torch

from core.problems.tsp.config import TSPConfig
from core.problems.tsp.sampler import TensorTSPSampler


def test_tensor_sampler_shape():
    config = TSPConfig(num_locations=10, distribution_of_locations="uniform")
    sampler = TensorTSPSampler(config)
    batch_size = 5

    sample = sampler(batch_size)

    assert sample.shape == (batch_size, config.num_locations, 2), (
        "Expected sample batch shape to be (batch_size, num_locations, 2)"
    )


def test_tensor_sampler_value_range():
    config = TSPConfig(
        num_locations=10,
        distribution_of_locations="uniform",
        min_coordinate=0.0,
        max_coordinate=1.0,
    )
    sampler = TensorTSPSampler(config)
    batch_size = 5

    sample = sampler(batch_size)

    assert torch.all(sample >= config.min_coordinate) and torch.all(
        sample <= config.max_coordinate
    ), "Expected all sample values to be within the specified coordinate range"


def test_tensor_sampler_determinism():
    config = TSPConfig(
        num_locations=10,
        distribution_of_locations="uniform",
        min_coordinate=0.0,
        max_coordinate=1.0,
    )
    seed = 42
    sampler1 = TensorTSPSampler(config, seed=seed)
    sampler2 = TensorTSPSampler(config, seed=seed)
    batch_size = 5

    sample1 = sampler1(batch_size)
    sample2 = sampler2(batch_size)

    assert torch.allclose(sample1, sample2), (
        "Expected samples from two samplers with the same seed to be identical"
    )


def test_tensor_sampler_different_seeds():
    config = TSPConfig(
        num_locations=10,
        distribution_of_locations="uniform",
        min_coordinate=0.0,
        max_coordinate=1.0,
    )
    seed1 = 42
    seed2 = 43
    sampler1 = TensorTSPSampler(config, seed=seed1)
    sampler2 = TensorTSPSampler(config, seed=seed2)
    batch_size = 5

    sample1 = sampler1(batch_size)
    sample2 = sampler2(batch_size)

    assert not torch.allclose(sample1, sample2), (
        "Expected samples from two samplers with different seeds to be different"
    )
