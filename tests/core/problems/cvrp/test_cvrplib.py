from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.problems.cvrp.dataset import (
    load_cvrplib_dataset,
    load_cvrplib_instance,
    load_cvrplib_solution_bks,
)


def _write_vrp(
    path: Path, coords: list[tuple[int, int]], demands: list[int], capacity: int
) -> None:
    """Write a minimal CVRPLIB-format .vrp file."""
    n = len(coords)
    lines = [
        "NAME : test",
        "TYPE : CVRP",
        f"DIMENSION : {n}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {capacity}",
        "NODE_COORD_SECTION",
    ]
    for i, (x, y) in enumerate(coords, start=1):
        lines.append(f"{i} {x} {y}")
    lines.append("DEMAND_SECTION")
    for i, d in enumerate(demands, start=1):
        lines.append(f"{i} {d}")
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")
    path.write_text("\n".join(lines))


def _write_sol(path: Path, routes: list[list[int]], cost: int) -> None:
    """Write a minimal CVRPLIB-format .sol file."""
    lines = []
    for i, route in enumerate(routes, start=1):
        lines.append(f"Route #{i}: {' '.join(str(c) for c in route)}")
    lines.append(f"Cost {cost}")
    path.write_text("\n".join(lines))


@pytest.fixture()
def tiny_vrp(tmp_path: Path) -> Path:
    """4-node instance: depot at (0,0), 3 customers at (300,0), (0,400), (300,400)."""
    path = tmp_path / "tiny.vrp"
    _write_vrp(
        path,
        coords=[(0, 0), (300, 0), (0, 400), (300, 400)],
        demands=[0, 30, 40, 50],
        capacity=100,
    )
    return path


def test_load_cvrplib_instance_shape(tiny_vrp: Path):
    instance = load_cvrplib_instance(tiny_vrp)

    assert instance.batch_size == torch.Size([])
    assert instance.num_nodes == 4  # depot + 3 customers
    assert instance.locations.shape == (4, 2)
    assert instance.demand.shape == (4,)
    assert instance.capacity.shape == (1,)
    assert instance.coordinate_scale.shape == (1,)


def test_load_cvrplib_instance_normalized(tiny_vrp: Path):
    instance = load_cvrplib_instance(tiny_vrp)

    # Coordinates are normalised in-place; demand/capacity are kept raw.
    assert instance.locations.max() <= 1.0 + 1e-6
    assert torch.all(instance.capacity == 100.0)
    assert torch.all(instance.demand[:1] == 0.0)  # depot demand is 0


def test_load_cvrplib_instance_demand_normalized(tiny_vrp: Path):
    instance = load_cvrplib_instance(tiny_vrp)

    assert torch.allclose(instance.demand[1:], torch.tensor([30.0, 40.0, 50.0]))
    assert torch.allclose(instance.normalized_demand[1:], torch.tensor([0.3, 0.4, 0.5]))


def test_load_cvrplib_instance_coordinate_scale(tiny_vrp: Path):
    instance = load_cvrplib_instance(tiny_vrp)

    # max coordinate across all nodes: max(300, 400, 300, 400) = 400
    assert torch.allclose(instance.coordinate_scale, torch.tensor([400.0]))


def test_load_cvrplib_dataset_returns_one_instance_per_file(tmp_path: Path):
    for name, coords in [
        ("a.vrp", [(0, 0), (100, 0), (0, 200)]),
        ("b.vrp", [(0, 0), (50, 50), (100, 100), (200, 200)]),
    ]:
        _write_vrp(
            tmp_path / name, coords=coords, demands=[0] + [10] * (len(coords) - 1), capacity=50
        )

    instances = load_cvrplib_dataset(tmp_path)

    assert len(instances) == 2
    assert instances[0].num_nodes == 3
    assert instances[1].num_nodes == 4
    assert instances[0].batch_size == torch.Size([])
    assert instances[1].batch_size == torch.Size([])


def test_load_cvrplib_dataset_sorted_by_name(tmp_path: Path):
    for name in ["c.vrp", "a.vrp", "b.vrp"]:
        _write_vrp(tmp_path / name, [(0, 0), (1, 0)], [0, 10], 50)

    instances = load_cvrplib_dataset(tmp_path)

    assert len(instances) == 3


def test_load_cvrplib_dataset_empty_directory_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No .vrp files"):
        load_cvrplib_dataset(tmp_path)


def test_load_cvrplib_solution_bks(tmp_path: Path):
    sol_path = tmp_path / "tiny.sol"
    _write_sol(sol_path, routes=[[2, 3], [4]], cost=1234)

    assert load_cvrplib_solution_bks(sol_path) == 1234
