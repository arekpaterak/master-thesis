import pytest
import torch

from core.problems.tsp.config import TSP
from core.problems.tsp.sampler import TSPSampler


@pytest.mark.parametrize(
    "batch_size",
    [1, 5, 10],
)
def test_sampler_shape(batch_size: int):
    config = TSP(num_locations=10, distribution_of_locations="uniform")
    sampler = TSPSampler(config)

    sample = sampler(batch_size)

    assert sample.batch_size[0] == batch_size, (
        "Expected sample batch to have the same batch size as requested"
    )
    assert sample.locations.shape == (batch_size, config.num_locations, 2), (
        "Expected sample locations to have shape (batch_size, num_locations, 2)"
    )


def test_sampler_value_range():
    config = TSP(
        num_locations=10,
        distribution_of_locations="uniform",
        min_coordinate=0.0,
        max_coordinate=1.0,
    )
    sampler = TSPSampler(config)
    batch_size = 5

    sample = sampler(batch_size)

    assert torch.all(sample.locations >= config.min_coordinate) and torch.all(
        sample.locations <= config.max_coordinate
    ), "Expected all sample values to be within the specified coordinate range"


def test_sampler_determinism():
    config = TSP(
        num_locations=10,
        distribution_of_locations="uniform",
        min_coordinate=0.0,
        max_coordinate=1.0,
    )
    seed = 42
    sampler1 = TSPSampler(config, seed=seed)
    sampler2 = TSPSampler(config, seed=seed)
    batch_size = 5

    sample1 = sampler1(batch_size)
    sample2 = sampler2(batch_size)

    assert torch.allclose(sample1.locations, sample2.locations), (
        "Expected samples from two samplers with the same seed to be identical"
    )


def test_sampler_different_seeds():
    config = TSP(
        num_locations=10,
        distribution_of_locations="uniform",
        min_coordinate=0.0,
        max_coordinate=1.0,
    )
    seed1 = 42
    seed2 = 43
    sampler1 = TSPSampler(config, seed=seed1)
    sampler2 = TSPSampler(config, seed=seed2)
    batch_size = 5

    sample1 = sampler1(batch_size)
    sample2 = sampler2(batch_size)

    assert not torch.allclose(sample1.locations, sample2.locations), (
        "Expected samples from two samplers with different seeds to be different"
    )


def test_access_single():
    config = TSP(
        num_locations=10,
    )
    sampler = TSPSampler(config)
    batch_size = 5

    sample = sampler(batch_size)

    single_sample = sample[0]

    assert single_sample.ndim == 0
    assert single_sample.locations.shape == (config.num_locations, 2), (
        "Expected single sample locations to have shape (num_locations, 2)"
    )
