import pytest
import torch

from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution


@pytest.fixture
def instance():
    config = CVRP(num_customers=5, vehicle_capacity_raw=10, min_demand_raw=1, max_demand_raw=1)
    return CVRPSampler(config, seed=0)(1)


def _make_solution(instance, sequence: list[list[int]]) -> CVRPGraphSolution:
    return CVRPGraphSolution.create_from_sequence(
        instance, torch.tensor(sequence, dtype=torch.long)
    )


def test_stitch_mid_route_removal(instance):
    """Removing a mid-route customer links its neighbours directly."""
    # Route: 0 -> 1 -> 2 -> 3 -> 0
    solution = _make_solution(instance, [[0, 1, 2, 3, 0, 4, 5, 0]])
    stitched = solution.remove_customers(torch.tensor([[2]]), stitch=True)

    # 1's next should now be 3 (skipping 2); 3's prev should be 1.
    assert stitched.next_nodes[0, 1].item() == 3
    assert stitched.prev_nodes[0, 3].item() == 1
    # Removed customer detached.
    assert stitched.next_nodes[0, 2].item() == CVRPGraphSolution.UNASSIGNED
    assert stitched.prev_nodes[0, 2].item() == CVRPGraphSolution.UNASSIGNED
    # Other route untouched.
    assert stitched.next_nodes[0, 4].item() == 5
    assert stitched.prev_nodes[0, 4].item() == 0


def test_stitch_adjacent_removal(instance):
    """Removing two consecutive customers stitches outer survivors directly."""
    # Route: 0 -> 1 -> 2 -> 3 -> 4 -> 0; remove 2 and 3.
    solution = _make_solution(instance, [[0, 1, 2, 3, 4, 0, 5, 0]])
    stitched = solution.remove_customers(torch.tensor([[2, 3]]), stitch=True)

    assert stitched.next_nodes[0, 1].item() == 4
    assert stitched.prev_nodes[0, 4].item() == 1
    for removed in (2, 3):
        assert stitched.next_nodes[0, removed].item() == CVRPGraphSolution.UNASSIGNED
        assert stitched.prev_nodes[0, removed].item() == CVRPGraphSolution.UNASSIGNED


def test_stitch_head_removal(instance):
    """Removing the route head promotes the next customer to head (prev = depot)."""
    solution = _make_solution(instance, [[0, 1, 2, 3, 0, 4, 5, 0]])
    stitched = solution.remove_customers(torch.tensor([[1]]), stitch=True)

    assert stitched.prev_nodes[0, 2].item() == CVRPGraphSolution.DEPOT
    assert stitched.next_nodes[0, 2].item() == 3
    assert stitched.next_nodes[0, 1].item() == CVRPGraphSolution.UNASSIGNED


def test_stitch_tail_removal(instance):
    """Removing the route tail makes the previous customer the new tail (next = depot)."""
    solution = _make_solution(instance, [[0, 1, 2, 3, 0, 4, 5, 0]])
    stitched = solution.remove_customers(torch.tensor([[3]]), stitch=True)

    assert stitched.next_nodes[0, 2].item() == CVRPGraphSolution.DEPOT
    assert stitched.prev_nodes[0, 2].item() == 1


def test_stitch_full_route_removal(instance):
    """Removing every customer in a route leaves no fragments on that route."""
    solution = _make_solution(instance, [[0, 1, 2, 3, 0, 4, 5, 0]])
    stitched = solution.remove_customers(torch.tensor([[1, 2, 3]]), stitch=True)

    for removed in (1, 2, 3):
        assert stitched.next_nodes[0, removed].item() == CVRPGraphSolution.UNASSIGNED
        assert stitched.prev_nodes[0, removed].item() == CVRPGraphSolution.UNASSIGNED
    # Other route untouched.
    assert stitched.next_nodes[0, 4].item() == 5
    assert stitched.prev_nodes[0, 4].item() == CVRPGraphSolution.DEPOT
    assert stitched.next_nodes[0, 5].item() == CVRPGraphSolution.DEPOT


def test_stitch_route_id_consistency(instance):
    """Survivors of an affected route share a single route_id after stitching."""
    solution = _make_solution(instance, [[0, 1, 2, 3, 4, 5, 0]])
    stitched = solution.remove_customers(torch.tensor([[2, 4]]), stitch=True)

    surviving = [1, 3, 5]
    route_ids = [stitched.route_ids[0, c].item() for c in surviving]
    assert len(set(route_ids)) == 1, f"Expected one shared route_id, got {route_ids}"


def test_stitch_preserves_completeness_for_survivors(instance):
    """Survivors have both prev and next set after stitching (no fragment endpoints)."""
    solution = _make_solution(instance, [[0, 1, 2, 3, 4, 5, 0]])
    stitched = solution.remove_customers(torch.tensor([[2, 4]]), stitch=True)

    is_removed = torch.zeros(instance.num_customers + 1, dtype=torch.bool)
    is_removed[2] = True
    is_removed[4] = True
    for c in range(1, instance.num_customers + 1):
        if is_removed[c]:
            continue
        assert stitched.prev_nodes[0, c].item() != CVRPGraphSolution.UNASSIGNED
        assert stitched.next_nodes[0, c].item() != CVRPGraphSolution.UNASSIGNED


def test_default_remains_fragmenting(instance):
    """Without stitch=True, neighbours of a removed customer become UNASSIGNED."""
    solution = _make_solution(instance, [[0, 1, 2, 3, 0, 4, 5, 0]])
    fragmented = solution.remove_customers(torch.tensor([[2]]))

    assert fragmented.next_nodes[0, 1].item() == CVRPGraphSolution.UNASSIGNED
    assert fragmented.prev_nodes[0, 3].item() == CVRPGraphSolution.UNASSIGNED
