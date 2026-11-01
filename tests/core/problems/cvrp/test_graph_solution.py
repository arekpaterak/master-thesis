import pytest
import torch

from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution


@pytest.fixture
def instance():
    config = CVRPDistribution(
        num_customers=4, vehicle_capacity_raw=10, min_demand_raw=1, max_demand_raw=1
    )
    return CVRPSampler(config, seed=0)(2)


@pytest.mark.parametrize(
    "sequence, nodes_to_remove",
    [
        # Standard contiguous routes
        ([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], None),
        # Partially deconstructed
        ([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], [[2], [3]]),
        ([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], [[1, 3], [2, 4]]),
        # All customers unassigned
        ([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], [[1, 2, 3, 4], [1, 2, 3, 4]]),
    ],
)
def test_valid_domain_and_depot_rules(instance, sequence, nodes_to_remove):
    sequence_tensor = torch.tensor(sequence, dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence_tensor)
    if nodes_to_remove is not None:
        solution = solution.remove_nodes(torch.tensor(nodes_to_remove, dtype=torch.long))

    N = instance.num_customers

    # Domain limits
    assert ((solution.next_nodes >= -1) & (solution.next_nodes <= N)).all()
    assert ((solution.prev_nodes >= -1) & (solution.prev_nodes <= N)).all()

    # Depot behaviour
    assert (solution.route_ids[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
    assert (solution.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
    assert (solution.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()

    # Customer constraints
    assert (solution.route_ids[..., 1:] >= 0).all()
    customer_indices = torch.arange(1, N + 1, device=solution.next_nodes.device).expand_as(
        solution.next_nodes[..., 1:]
    )
    assert (solution.next_nodes[..., 1:] != customer_indices).all()
    assert (solution.prev_nodes[..., 1:] != customer_indices).all()


@pytest.mark.parametrize(
    "sequence, nodes_to_remove",
    [
        ([[0, 1, 2, 3, 4, 0], [0, 4, 3, 2, 1, 0]], None),
        ([[0, 1, 2, 3, 4, 0], [0, 4, 3, 2, 1, 0]], [[2], [3]]),
        ([[0, 1, 2, 3, 4, 0], [0, 4, 3, 2, 1, 0]], [[2, 3], [1, 4]]),
    ],
)
def test_bi_directional_consistency(instance, sequence, nodes_to_remove):
    sequence_tensor = torch.tensor(sequence, dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence_tensor)
    if nodes_to_remove is not None:
        solution = solution.remove_nodes(torch.tensor(nodes_to_remove, dtype=torch.long))

    N = instance.num_customers
    device = solution.next_nodes.device
    node_indices = torch.arange(N + 1, device=device).expand_as(solution.next_nodes)

    # Forward: If i points to j (and j > 0), j must point back to i in prev
    valid_next_mask = solution.next_nodes > 0
    assert (
        solution.prev_nodes.gather(-1, solution.next_nodes.clamp(min=0))[valid_next_mask]
        == node_indices[valid_next_mask]
    ).all()

    # Backward: If j is preceded by i (and i > 0), i must point forward to j
    valid_prev_mask = solution.prev_nodes > 0
    assert (
        solution.next_nodes.gather(-1, solution.prev_nodes.clamp(min=0))[valid_prev_mask]
        == node_indices[valid_prev_mask]
    ).all()


@pytest.mark.parametrize(
    "sequence, nodes_to_remove",
    [
        ([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], None),
        ([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], [[2], [3]]),
    ],
)
def test_route_id_consistency_along_edges(instance, sequence, nodes_to_remove):
    sequence_tensor = torch.tensor(sequence, dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence_tensor)
    if nodes_to_remove is not None:
        solution = solution.remove_nodes(torch.tensor(nodes_to_remove, dtype=torch.long))

    N = instance.num_customers
    node_indices = torch.arange(N + 1, device=solution.next_nodes.device).expand_as(
        solution.next_nodes
    )

    # Only test connected customers (both nodes > 0)
    customer_to_customer_mask = (node_indices > 0) & (solution.next_nodes > 0)

    route_id_A = solution.route_ids[customer_to_customer_mask]
    route_id_B = solution.route_ids.gather(-1, solution.next_nodes.clamp(min=0))[
        customer_to_customer_mask
    ]

    assert (route_id_A == route_id_B).all()


def test_one_customer_per_route_invariants(instance):
    solution = CVRPGraphSolution.one_customer_per_route(instance)

    N = instance.num_customers

    # Depot points nowhere in CVRPGraphSolution one_customer_per_route
    assert (solution.next_nodes[..., 1:] == CVRPGraphSolution.DEPOT).all()
    assert (solution.prev_nodes[..., 1:] == CVRPGraphSolution.DEPOT).all()

    assert (solution.route_ids[..., 0] == CVRPGraphSolution.UNASSIGNED).all()

    # Ensure route IDs are unique and strictly >= 0 for customers
    unique_customer_routes = solution.route_ids[..., 1:]
    assert (unique_customer_routes >= 0).all()

    # Ensure no customer shares a route ID
    for b in range(instance.batch_size[0]):
        assert len(torch.unique(unique_customer_routes[b])) == N


def test_route_id_equals_head_index(instance):
    """After create_from_sequence, route_id[i] == head node index for all customers."""
    # Route 1: depot -> 1 -> 2 -> depot. Head is 1 (prev[1] = 0).
    # Route 2: depot -> 3 -> 4 -> depot. Head is 3 (prev[3] = 0).
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)

    # Batch 0: customers 1 and 2 share route, head = 1
    assert solution.route_ids[0, 1] == 1
    assert solution.route_ids[0, 2] == 1
    # Customers 3 and 4 share route, head = 3
    assert solution.route_ids[0, 3] == 3
    assert solution.route_ids[0, 4] == 3

    # Batch 1: customers 4 and 3 share route, head = 3 (prev[3] = 0 after scatter)
    # (sequence is [0, 4, 3, 0, 2, 1, 0] => _prev[4]=0, _prev[3]=4>0 => head is 4)
    assert solution.route_ids[1, 4] == 4
    assert solution.route_ids[1, 3] == 4
    assert solution.route_ids[1, 2] == 2
    assert solution.route_ids[1, 1] == 2


def test_isolated_customer_is_own_route_head(instance):
    """After remove, isolated customer's route_id equals its own index."""
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)
    solution = solution.remove_nodes(torch.tensor([[2], [3]], dtype=torch.long))

    # Customer 2 in batch 0 is now isolated; route_id should be 2
    assert solution.route_ids[0, 2] == 2
    # Customer 3 in batch 1 is now isolated; route_id should be 3
    assert solution.route_ids[1, 3] == 3


def test_objective_matches_sequence_solution(instance):
    """Graph objective must equal sequence objective on the same routes."""
    from core.problems.cvrp.solution import CVRPSolution

    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    graph_solution = CVRPGraphSolution.create_from_sequence(instance, sequence)
    sequence_solution = CVRPSolution.create(instance=instance, sequence=sequence)

    torch.testing.assert_close(graph_solution.objective(), sequence_solution.objective())


def test_num_routes_after_create(instance):
    """create_from_sequence: num_routes equals the number of depot-separated segments."""
    # Two routes in each batch element
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)
    assert (solution.num_routes == 2).all()


def test_num_routes_increases_after_remove(instance):
    """Removing a mid-route customer splits the route, increasing num_routes."""
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 1, 2, 0, 3, 4, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)
    # Remove a middle customer to split route [1,2] into [1] and [2]
    solution = solution.remove_nodes(torch.tensor([[2], [2]], dtype=torch.long))
    # Was 2 routes; after splitting one route and isolating customer 2: 3 routes each
    assert (solution.num_routes == 3).all()


def test_num_routes_one_customer_per_route(instance):
    """one_customer_per_route: num_routes == num_customers."""
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    assert (solution.num_routes == instance.num_customers).all()


def test_remove_route_head_preserves_depot_slot(instance):
    """Removing a route head (predecessor is depot) must not corrupt _next[0]."""
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)
    # Customer 1 in batch 0 is a route head (prev[1] = 0)
    solution = solution.remove_nodes(torch.tensor([[1], [4]], dtype=torch.long))

    assert (solution.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
    assert (solution.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()


def test_remove_route_tail_preserves_depot_slot(instance):
    """Removing a route tail (successor is depot) must not corrupt _prev[0]."""
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)
    # Customer 2 in batch 0 is a route tail (next[2] = 0)
    solution = solution.remove_nodes(torch.tensor([[2], [3]], dtype=torch.long))

    assert (solution.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
    assert (solution.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()


def test_add_edges_sets_pointers(instance):
    """add_edges: writes _next[u] = v and _prev[v] = u."""
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    # All customers have next/prev = DEPOT (0). Remove two to get UNASSIGNED tails.
    solution = solution.remove_nodes(torch.tensor([[1, 2], [3, 4]], dtype=torch.long))

    u = torch.tensor([1, 3], dtype=torch.long)  # tails (next == UNASSIGNED)
    v = torch.tensor([2, 4], dtype=torch.long)  # heads (prev == UNASSIGNED)
    connected = solution.add_edges(torch.stack([u, v], dim=-1).unsqueeze(-2))

    assert (connected.next_nodes[0, 1] == 2).all()
    assert (connected.prev_nodes[0, 2] == 1).all()
    assert (connected.next_nodes[1, 3] == 4).all()
    assert (connected.prev_nodes[1, 4] == 3).all()


def test_add_edges_route_id_merge_uses_minimum(instance):
    """add_edges: merged route gets the minimum of the two route IDs."""
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    solution = solution.remove_nodes(torch.tensor([[1, 2], [1, 2]], dtype=torch.long))
    # In one_customer_per_route, route_id[i] = i. After remove, still same (own-head).
    # Connect customer 3 (route_id=3) -> customer 1 (route_id=1): merged = min(3,1) = 1.
    u = torch.tensor([3, 3], dtype=torch.long)
    v = torch.tensor([1, 1], dtype=torch.long)
    connected = solution.add_edges(torch.stack([u, v], dim=-1).unsqueeze(-2))

    assert (connected.route_ids[..., 3] == 1).all()
    assert (connected.route_ids[..., 1] == 1).all()


def test_add_edges_preserves_depot_invariant(instance):
    """add_edges must not modify the depot slot in _next or _prev."""
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    solution = solution.remove_nodes(torch.tensor([[1, 2], [3, 4]], dtype=torch.long))

    u = torch.tensor([1, 3], dtype=torch.long)
    v = torch.tensor([2, 4], dtype=torch.long)
    connected = solution.add_edges(torch.stack([u, v], dim=-1).unsqueeze(-2))

    assert (connected.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
    assert (connected.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()


def test_add_edges_with_depot_preserves_depot_slot(instance):
    """Edges involving the depot (u=0 or v=0) must not corrupt _next[0] or _prev[0]."""
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    solution = solution.remove_nodes(torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long))

    # Rebuild route: depot->1->2->3->depot, plus leave customer 4 as-is (depot->4->depot).
    edges = torch.tensor(
        [
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 0]],
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 0]],
        ],
        dtype=torch.long,
    )
    connected = solution.add_edges(edges)

    assert (connected.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
    assert (connected.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()

    # Pointer chain for the rebuilt route
    assert (connected.next_nodes[:, 1] == 2).all()
    assert (connected.next_nodes[:, 2] == 3).all()
    assert (connected.next_nodes[:, 3] == 0).all()
    assert (connected.prev_nodes[:, 1] == 0).all()
    assert (connected.prev_nodes[:, 2] == 1).all()
    assert (connected.prev_nodes[:, 3] == 2).all()

    assert connected.is_complete.all()


def test_demand_per_route_conservation(instance):
    """Total demand across all routes must equal total customer demand."""
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)

    total_by_route = solution.demand_per_route().sum(dim=-1)
    total_customers = instance.demand[..., 1:].sum(dim=-1)
    torch.testing.assert_close(total_by_route, total_customers)


def test_demand_per_route_after_remove_conservation(instance):
    """Conservation must hold even with isolated (removed) customers."""
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)
    solution = solution.remove_nodes(torch.tensor([[2], [3]], dtype=torch.long))

    total_by_route = solution.demand_per_route().sum(dim=-1)
    total_customers = instance.demand[..., 1:].sum(dim=-1)
    torch.testing.assert_close(total_by_route, total_customers)


def test_is_feasible_complete_within_capacity(instance):
    """A complete solution where each route demand <= capacity is feasible."""
    # With all-demand-1 fixture and capacity 10, any grouping is feasible.
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    # one_customer_per_route gives every customer its own route - trivially feasible.
    # But num_unassigned > 0 because _next[i] = DEPOT (not UNASSIGNED), so is_complete = True.
    assert solution.is_complete.all()
    assert solution.is_feasible.all()


def test_is_not_complete_after_remove(instance):
    """After remove_customers, solution is not complete."""
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)
    solution = solution.remove_nodes(torch.tensor([[1], [1]], dtype=torch.long))
    assert not solution.is_complete.any()


def test_multi_dim_batch_shape(instance):
    """All operations must work with (B, R) batch shapes from rollouts."""
    from core.problems.cvrp import CVRPDistribution, CVRPSampler

    config = CVRPDistribution(
        num_customers=4, vehicle_capacity_raw=10, min_demand_raw=1, max_demand_raw=1
    )
    instance = CVRPSampler(config, seed=0)(2)

    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0]], dtype=torch.long)
    base = CVRPGraphSolution.create_from_sequence(instance, sequence)

    # Expand to (2, 3) batch shape as a rollout would
    expanded = base.unsqueeze(-1).expand(2, 3)
    assert expanded.batch_size == torch.Size([2, 3])

    # remove_customers with (2, 3, M) customers tensor
    customers = torch.ones(2, 3, 1, dtype=torch.long)
    result = expanded.remove_nodes(customers)
    assert result.batch_size == torch.Size([2, 3])
    assert (result.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
    assert (result.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
