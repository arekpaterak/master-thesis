import pickle
from pathlib import Path

import pytest
import torch

from core.problems.cvrp.dataset import load_cvrp_dataset
from core.problems.dataset import DatasetSampler


def _write_cvrp_pickle(path: Path, num_instances: int, num_customers: int = 3) -> None:
    """Write a tiny benchmark-format CVRP pickle.

    Each instance encodes its index in the depot x-coordinate so slices can be
    checked for identity and ordering.
    """
    dataset = []
    for index in range(num_instances):
        depot = [float(index), 0.0]
        locations = [[0.1 * (c + 1), 0.2 * (c + 1)] for c in range(num_customers)]
        demand = [c + 1 for c in range(num_customers)]
        capacity = 10.0
        dataset.append((depot, locations, demand, capacity))
    with open(path, "wb") as f:
        pickle.dump(dataset, f)


def test_load_cvrp_dataset_shape_and_normalization(tmp_path: Path):
    path = tmp_path / "cvrp.pkl"
    _write_cvrp_pickle(path, num_instances=4, num_customers=3)

    instance = load_cvrp_dataset(path)

    assert instance.batch_size[0] == 4
    assert instance.num_nodes == 4  # 3 customers + depot
    assert instance.locations.shape == (4, 4, 2)
    assert instance.demand.shape == (4, 4)
    assert instance.capacity.shape == (4, 1)

    # Depot is prepended; demand and capacity are kept raw (capacity 10.0).
    assert torch.allclose(instance.locations[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.all(instance.demand[:, 0] == 0.0)
    assert torch.allclose(instance.demand[0, 1:], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.all(instance.capacity == 10.0)
    # Normalisation is derived on the fly for the network.
    assert torch.allclose(instance.normalized_demand[0, 1:], torch.tensor([0.1, 0.2, 0.3]))


def test_load_cvrp_dataset_inconsistent_size_raises(tmp_path: Path):
    path = tmp_path / "ragged.pkl"
    dataset = [
        ([0.0, 0.0], [[0.1, 0.1], [0.2, 0.2]], [1, 2], 10.0),
        ([0.0, 0.0], [[0.1, 0.1]], [1], 10.0),
    ]
    with open(path, "wb") as f:
        pickle.dump(dataset, f)

    with pytest.raises(ValueError, match="inconsistent number of nodes"):
        load_cvrp_dataset(path)


def test_dataset_sampler_serves_contiguous_slices(tmp_path: Path):
    path = tmp_path / "cvrp.pkl"
    _write_cvrp_pickle(path, num_instances=10)
    instance = load_cvrp_dataset(path)
    sampler = DatasetSampler(instance)

    first = sampler(4)
    second = sampler(4)

    assert first.batch_size[0] == 4
    assert second.batch_size[0] == 4
    # Depot x-coordinate encodes the instance index, so slices must be ordered.
    assert first.locations[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert second.locations[:, 0, 0].tolist() == [4.0, 5.0, 6.0, 7.0]


def test_dataset_sampler_reset_replays_same_instances(tmp_path: Path):
    path = tmp_path / "cvrp.pkl"
    _write_cvrp_pickle(path, num_instances=10)
    sampler = DatasetSampler(load_cvrp_dataset(path))

    first_pass = sampler(5)
    sampler.reset()
    second_pass = sampler(5)

    assert torch.equal(first_pass.locations, second_pass.locations)
    assert first_pass.locations[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_dataset_sampler_wraps_around(tmp_path: Path):
    path = tmp_path / "cvrp.pkl"
    _write_cvrp_pickle(path, num_instances=6)
    sampler = DatasetSampler(load_cvrp_dataset(path))

    sampler(4)  # cursor -> 4
    wrapped = sampler(4)  # 4, 5, then wraps to 0, 1

    assert wrapped.locations[:, 0, 0].tolist() == [4.0, 5.0, 0.0, 1.0]
    assert sampler._cursor == 2


def test_dataset_sampler_rejects_oversized_batch(tmp_path: Path):
    path = tmp_path / "cvrp.pkl"
    _write_cvrp_pickle(path, num_instances=3)
    sampler = DatasetSampler(load_cvrp_dataset(path))

    with pytest.raises(ValueError, match="exceeds dataset size"):
        sampler(4)
