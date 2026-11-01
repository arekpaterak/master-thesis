import pytest
import torch

from core.problems.cvrp import CVRPDistribution, CVRPSampler, CVRPSolution
from core.problems.cvrp.solve import all_customers_one_route, one_customer_per_route


def test_referencing_to_instance():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)

    assert solution.instance is instance

    # The underlying tensors should be shared
    assert solution.instance.locations.data_ptr() == instance.locations.data_ptr()
    assert solution.instance.demand.data_ptr() == instance.demand.data_ptr()
    assert solution.instance.capacity.data_ptr() == instance.capacity.data_ptr()


def test_solution_clone_does_not_alias_instance_tensors():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(2)
    solution = one_customer_per_route(instance)

    cloned = solution.clone()

    # clone() should produce a different instance object
    assert cloned is not solution
    assert cloned.instance is not solution.instance

    # Modifications to the clone should not affect the original
    cloned.sequence[0, 1] = (cloned.sequence[0, 1] + 1).clamp(min=0)
    assert not torch.equal(cloned.sequence, solution.sequence)


def test_solution_to_device_moves_instance_and_sequence():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(3)
    solution = one_customer_per_route(instance)

    moved = solution.to("cpu")
    assert moved.sequence.device.type == "cpu"
    assert moved.instance.locations.device.type == "cpu"
    assert moved.instance.demand.device.type == "cpu"
    assert moved.instance.capacity.device.type == "cpu"

    if torch.cuda.is_available():
        moved_cuda = solution.to("cuda")
        assert moved_cuda.sequence.device.type == "cuda"
        assert moved_cuda.instance.locations.device.type == "cuda"
        assert moved_cuda.instance.demand.device.type == "cuda"
        assert moved_cuda.instance.capacity.device.type == "cuda"


def test_instance_to_device_moves_solution_instance():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(3)
    solution = one_customer_per_route(instance)

    instance.to("cpu")
    assert solution.instance is instance

    if torch.cuda.is_available():
        instance.to("cuda", inplace=True)
        assert solution.instance is instance
        assert instance.locations.device.type == "cuda"
        assert solution.instance.locations.device.type == "cuda"
        assert solution.instance.demand.device.type == "cuda"
        assert solution.instance.capacity.device.type == "cuda"


def test_solution_slices_keep_per_element_instance_reference():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)

    head = solution[:1]
    assert head.batch_size.numel() == 1
    assert head.instance.batch_size.numel() == 1

    assert torch.allclose(head.instance.locations, solution.instance.locations[:1])
    assert torch.equal(head.sequence, solution.sequence[:1])

    tail = solution[1:]
    assert tail.batch_size.numel() == 9
    assert tail.instance.batch_size.numel() == 9

    assert torch.allclose(tail.instance.locations, solution.instance.locations[1:])
    assert torch.equal(tail.sequence, solution.sequence[1:])


def test_concatenate_two_solutions():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution1 = one_customer_per_route(instance)
    solution2 = one_customer_per_route(instance)

    combined_solution: CVRPSolution = torch.cat([solution1, solution2])
    assert combined_solution.batch_size.numel() == 20

    assert combined_solution.instance.locations.shape[0] == 20
    assert torch.allclose(combined_solution.instance.locations[:10], instance.locations)
    assert torch.allclose(combined_solution.instance.locations[10:], instance.locations)


def test_if_can_visualize_without_batch_dim():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)[0]
    assert solution.batch_dims == 0

    solution.visualize()


def test_if_can_visualize_with_batch_of_one():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)[:1]
    assert solution.batch_dims == 1

    solution.visualize()


def test_if_fails_to_visualize_batch():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)

    with pytest.raises(ValueError):
        solution.visualize()


def test_single_instance_routes():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)[0]

    routes = solution.routes()
    assert len(routes) == instance.num_customers
    for route in routes:
        assert len(route) == 1


def test_objective():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)

    objective = solution.objective()
    assert (objective > 0).all()


def test_total_distance_rounded_false_matches_objective():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(4)
    solution = one_customer_per_route(instance)

    assert torch.allclose(solution.total_distance(rounded=False), solution.objective())


def test_total_distance_rounded_recovers_original_scale():
    from core.problems.cvrp.instance import CVRPInstance

    # Two nodes: depot at (0,0), one customer at (300, 400) - in original integer coords.
    # normalize_() scales by amax = 400, so locations become [[0,0],[0.75,1.0]].
    # Distance per leg = sqrt(0.75^2 + 1^2) = 1.25 (normalized), 500 (original).
    locations = torch.tensor([[[0.0, 0.0], [300.0, 400.0]]])  # (1, 2, 2)
    demand = torch.tensor([[0.0, 0.5]])
    capacity = torch.ones(1, 1)
    instance = CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)
    instance.normalize_()  # scale = 400

    # Sequence: depot -> customer -> depot
    sequence = torch.tensor([[0, 1, 0]])
    solution = CVRPSolution.create(instance=instance, sequence=sequence)

    dist_normalized = solution.total_distance(rounded=False)
    dist_rounded = solution.total_distance(rounded=True)

    # Normalized: 2 * sqrt(0.75^2 + 1^2) = 2 * 1.25 = 2.5
    assert torch.allclose(dist_normalized, torch.tensor([2.5]))
    # Original-scale rounded: 2 * round(sqrt(300^2 + 400^2)) = 2 * 500 = 1000
    assert torch.allclose(dist_rounded, torch.tensor([1000.0]))


def test_mean_objective_per_instance_in_batch():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)
    rollout_size = 5
    expanded_solution = solution.unsqueeze(-1).expand(*solution.batch_size, rollout_size)

    objective = expanded_solution.objective()
    mean_objective_per_instance_in_batch = objective.mean(dim=1)

    assert mean_objective_per_instance_in_batch.shape == (10,)
    assert torch.allclose(
        mean_objective_per_instance_in_batch,
        solution.objective(),
    )


def test_gather_from_multiple_rollouts():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)
    rollout_size = 5
    expanded_solution = solution.unsqueeze(-1).expand(*solution.batch_size, rollout_size)
    best_per_instance = torch.randint(0, rollout_size, (10,))

    gathered_solution = expanded_solution.gather(1, best_per_instance).squeeze(1)

    assert gathered_solution.batch_dims == 1
    assert gathered_solution.instance.batch_size == (10,)


def test_is_complete():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)

    # sequence: [0, 1, 0, 2, 0, ..., 0, 19, 0], expected: True
    solution = one_customer_per_route(instance)
    assert (solution.is_complete == True).all()

    # sequence: [0, 1, 2, ..., 19, 0], expected: False
    solution.sequence = solution.sequence[:, :-3]
    assert (solution.is_complete == False).all()


def test_deconstructed():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)

    solution.sequence[0, 4] = -1

    assert (solution.is_deconstructed == True)[0]
    assert (solution.is_complete == False)[0]
    assert (solution.is_complete == True)[1:].all()

    solution.sequence[:, 4] = -1

    assert (solution.is_deconstructed == True).all()
    assert (solution.is_complete == False).all()


def test_is_feasible():
    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)
    assert solution.is_feasible.all()

    infeasible_config = CVRPDistribution(
        num_customers=20,
        vehicle_capacity_raw=1,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    infeasible_instance = CVRPSampler(infeasible_config, seed=0)(10)
    infeasible_solution = all_customers_one_route(infeasible_instance)
    assert (~infeasible_solution.is_feasible).all()

    feasible_config = CVRPDistribution(
        num_customers=20,
        vehicle_capacity_raw=20,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    feasible_instance = CVRPSampler(feasible_config, seed=0)(10)
    feasible_solution = all_customers_one_route(feasible_instance)
    assert feasible_solution.is_feasible.all()

    combined_solution: CVRPSolution = torch.cat([feasible_solution, infeasible_solution], dim=0)
    print(combined_solution)

    assert combined_solution.is_feasible.any() and (~combined_solution.is_feasible).any()


def test_is_feasible_on_deconstructed():
    """
    Test if is_feasible is safe to call on deconstructed solutions.
    """

    instance = CVRPSampler(CVRPDistribution.for_n(20))(10)
    solution = one_customer_per_route(instance)

    # Deconstruct one element by introducing an UNASSIGNED token.
    solution.sequence[0, 4] = -1
    assert solution.is_deconstructed[0]

    # Must not raise
    feasible = solution.is_feasible

    assert feasible.dtype == torch.bool
    assert feasible.shape == (10,)
    assert feasible[0].item() is False

    # Ensure the masking behaviour is per-batch-element and is not all-or-nothing.
    assert feasible[1:].all().item() is True


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ([0], [0]),
        ([0, 0], [0, 0]),
        ([0, 1, 2, 3, 0], [0, 0, 0, 0, 0]),
        ([0, 1, 0, 2, 0, 3, 0], [0, 0, 1, 1, 2, 2, 2]),
        ([0, 1, 0], [0, 0, 0]),
        ([0, 1, 0, 2, 0], [0, 0, 1, 1, 1]),
        ([0, 1, -1, 0], [0, 0, 0, 0]),
        ([0, 1, -1, 0, 3, 0], [0, 0, 0, 1, 1, 1]),
        ([0, 1, -1, 3, 0], [0, 0, 1, 1, 1]),
        ([0, -1, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1]),
        ([0, 1, -1, -1, 2, 0], [0, 0, 0, 1, 1, 1]),
    ],
)
def test_route_indexed_sequence(sequence, expected):
    sequence = torch.tensor(sequence, dtype=torch.long)
    expected = torch.tensor(expected, dtype=torch.long)
    instance = CVRPSampler(CVRPDistribution.for_n(20))(1)
    solution = CVRPSolution.create(
        instance=instance,
        sequence=sequence.unsqueeze(0),
    )

    route_indexed_sequence = solution.route_indexed_sequence

    assert torch.equal(route_indexed_sequence[0], expected)


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ([0, 0], 1),
        ([0, 1, 2, 3, 0], 1),
        ([0, 1, 0, 2, 0, 3, 0], 3),
        ([0, 1, 0], 1),
        ([0, 1, 0, 2, 0], 2),
        ([0, 1, -1, 0], 1),
        ([0, 1, -1, 0, 3, 0], 2),
        ([0, 1, -1, 3, 0], 2),
        ([0, -1, 0, -1, 0, -1, 0], 1),
    ],
)
def test_num_routes(sequence, expected):
    sequence = torch.tensor(sequence, dtype=torch.long)
    instance = CVRPSampler(CVRPDistribution.for_n(20))(1)
    solution = CVRPSolution.create(
        instance=instance,
        sequence=sequence.unsqueeze(0),
    )

    num_routes = solution.num_routes[0]

    assert num_routes == expected


# demand is stored raw now (demand=1 per customer, capacity=100), so demand_per_route
# returns raw route loads (one per customer in the route), not capacity-normalised ones.
@pytest.mark.parametrize(
    "sequence,expected",
    [
        ([0], [0]),
        ([0, 0], [0]),
        ([0, 1, 2, 3, 0], [3]),
        ([0, 1, 0, 2, 0, 3, 0], [1, 1, 1]),
        ([0, 1, 0], [1]),
        ([0, 1, 0, 2, 0], [1, 1]),
        ([0, 1, -1, 0], [1]),
        ([0, 1, -1, 0, 3, 0], [1, 1]),
        ([0, 1, -1, 3, 0], [1, 1]),
        ([0, -1, 0, -1, 0, -1, 0], [0]),
        ([0, 1, 0, 0, 2, 0, 0, 0], [1, 1]),
        ([0, 1, -1, -1, 2, 0], [1, 1]),
    ],
)
def test_demand_per_route(sequence, expected):
    sequence = torch.tensor(sequence, dtype=torch.long)
    expected = torch.tensor(expected, dtype=torch.float32)
    config = CVRPDistribution(
        num_customers=100,
        vehicle_capacity_raw=100,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(1)
    solution = CVRPSolution.create(
        instance=instance,
        sequence=sequence.unsqueeze(0),
    )

    demand_per_route = solution.demand_per_route()

    assert torch.allclose(demand_per_route[0], expected)


@pytest.mark.parametrize(
    "sequence,expected",
    [
        (
            [[0, 1, 0, 2, 0, 3, 0, 0, 0], [0, 1, 0, 2, 0, 3, 0, 4, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 1]],
        ),
    ],
)
def test_demand_per_route_with_mixed_num_routes_in_batch(sequence, expected):
    sequence = torch.tensor(sequence, dtype=torch.long)
    expected = torch.tensor(expected, dtype=torch.float32)
    config = CVRPDistribution(
        num_customers=100,
        vehicle_capacity_raw=100,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(2)
    solution = CVRPSolution.create(
        instance=instance,
        sequence=sequence,
    )

    demand_per_route = solution.demand_per_route()

    assert torch.allclose(demand_per_route, expected)


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ([0], [-1, -1, -1]),
        ([0, 0], [-1, -1, -1]),
        ([0, 1, 2, 3, 0], [0, 0, 0]),
        ([0, 1, 0, 2, 0, 3, 0], [0, 1, 2]),
        ([0, 1, 0], [0, -1, -1]),
        ([0, 1, 0, 2, 0], [0, 1, -1]),
        ([0, 1, -1, 0], [0, -1, -1]),
        ([0, 1, -1, 0, 3, 0], [0, -1, 1]),
        ([0, 1, -1, 3, 0], [0, -1, 1]),
        ([0, -1, 0, -1, 0, -1, 0], [-1, -1, -1]),
        ([0, 1, 0, 0, 2, 0, 0, 0], [0, 1, -1]),
        ([0, 1, -1, -1, 2, 0], [0, 1, -1]),
    ],
)
def test_route_per_customer(sequence, expected):
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
    expected = torch.tensor(expected, dtype=torch.long).unsqueeze(0)
    config = CVRPDistribution(
        num_customers=3,
        vehicle_capacity_raw=3,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(1)
    solution = CVRPSolution.create(
        instance=instance,
        sequence=sequence,
    )

    route_per_customer = solution.route_per_customer()

    assert torch.equal(route_per_customer, expected)


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ([0], [[-1, -1], [-1, -1], [-1, -1]]),
        ([0, 0], [[-1, -1], [-1, -1], [-1, -1]]),
        ([0, 1, 2, 3, 0], [[0, 2], [1, 3], [2, 0]]),
        ([0, 1, 0, 2, 0, 3, 0], [[0, 0], [0, 0], [0, 0]]),
        ([0, 1, 0], [[0, 0], [-1, -1], [-1, -1]]),
        ([0, 1, 0, 2, 0], [[0, 0], [0, 0], [-1, -1]]),
        ([0, 1, -1, 0], [[0, -1], [-1, -1], [-1, -1]]),
        ([0, 1, -1, 0, 3, 0], [[0, -1], [-1, -1], [0, 0]]),
        ([0, 1, -1, 3, 0], [[0, -1], [-1, -1], [-1, 0]]),
        ([0, -1, 0, -1, 0, -1, 0], [[-1, -1], [-1, -1], [-1, -1]]),
        ([0, 1, 0, 0, 2, 0, 0, 0], [[0, 0], [0, 0], [-1, -1]]),
        ([0, 1, -1, -1, 2, 0], [[0, -1], [-1, 0], [-1, -1]]),
    ],
)
def test_neighbours_per_customer(sequence, expected):
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
    expected = torch.tensor(expected, dtype=torch.long).unsqueeze(0)
    config = CVRPDistribution(
        num_customers=3,
        vehicle_capacity_raw=3,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(1)
    solution = CVRPSolution.create(
        instance=instance,
        sequence=sequence,
    )

    neighbours_per_customer = solution.neighbours_per_customer()

    assert torch.equal(neighbours_per_customer, expected)


@pytest.mark.parametrize(
    "sequence,to_remove,expected",
    [
        ([[0, 1, 2, 3, 0]], [[[1, 3], [2, 1]]], [[[0, -1, 2, -1, 0], [0, -1, -1, 3, 0]]]),
        ([[0, 1, 2, 3, 0]], [[1]], [[0, -1, 2, 3, 0]]),
        (
            [[0, 1, 2, 0]],
            [[0, 1]],  # Trying to remove depot (0) and customer (1)
            [[0, -1, 2, 0]],  # Depot should remain 0
        ),
        ([[0, -1, 2, 3, 0]], [[1, 2, 4]], [[0, -1, -1, 3, 0]]),
    ],
)
def test_remove_customers(sequence, to_remove, expected):
    sequence = torch.tensor(sequence, dtype=torch.long)
    to_remove = torch.tensor(to_remove, dtype=torch.long)
    expected = torch.tensor(expected, dtype=torch.long)
    config = CVRPDistribution(
        num_customers=10,
        vehicle_capacity_raw=10,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(sequence.shape[0])
    solution = CVRPSolution.create(instance=instance, sequence=sequence)

    deconstructed_solution = solution.remove_customers(to_remove)

    assert torch.equal(deconstructed_solution.sequence, expected)
    assert deconstructed_solution.is_deconstructed.all()
    assert deconstructed_solution.instance.batch_size == deconstructed_solution.batch_size
    assert deconstructed_solution.instance.locations.shape[:-2] == deconstructed_solution.batch_size


def test_remove_nodes_rejects_stitch():
    solution = one_customer_per_route(CVRPSampler(CVRPDistribution.for_n(20))(2))
    with pytest.raises(ValueError, match="CVRPGraphSolution"):
        solution.remove_nodes(torch.tensor([[1], [2]]), stitch=True)
