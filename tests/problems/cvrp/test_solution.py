import pytest
import torch

from core.models.nds.preprocessing import build_encoder_input_indexes
from core.problems.cvrp import CVRPConfig, CVRPSampler, CVRPSolution
from core.problems.cvrp.solve import all_customers_one_route, one_customer_per_route


def test_referencing_to_instance():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)

    assert solution.instance is instance

    # The underlying tensors should be shared
    assert solution.instance.locations.data_ptr() == instance.locations.data_ptr()
    assert solution.instance.demand.data_ptr() == instance.demand.data_ptr()
    assert solution.instance.capacity.data_ptr() == instance.capacity.data_ptr()


def test_solution_clone_does_not_alias_instance_tensors():
    instance = CVRPSampler(CVRPConfig.for_n(20))(2)
    solution = one_customer_per_route(instance)

    cloned = solution.clone()

    # clone() should produce a different instance object
    assert cloned is not solution
    assert cloned.instance is not solution.instance

    # Modifications to the clone should not affect the original
    cloned.sequence[0, 1] = (cloned.sequence[0, 1] + 1).clamp(min=0)
    assert not torch.equal(cloned.sequence, solution.sequence)


def test_solution_to_device_moves_instance_and_sequence():
    instance = CVRPSampler(CVRPConfig.for_n(20))(3)
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
    instance = CVRPSampler(CVRPConfig.for_n(20))(3)
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
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
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
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution1 = one_customer_per_route(instance)
    solution2 = one_customer_per_route(instance)

    combined_solution: CVRPSolution = torch.cat([solution1, solution2])
    assert combined_solution.batch_size.numel() == 20

    assert combined_solution.instance.locations.shape[0] == 20
    assert torch.allclose(combined_solution.instance.locations[:10], instance.locations)
    assert torch.allclose(combined_solution.instance.locations[10:], instance.locations)


def test_if_can_visualize_without_batch_dim():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)[0]
    assert solution.batch_dims == 0

    solution.visualize()


def test_if_can_visualize_with_batch_of_one():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)[:1]
    assert solution.batch_dims == 1

    solution.visualize()


def test_if_fails_to_visualize_batch():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)

    with pytest.raises(ValueError):
        solution.visualize()


def test_single_instance_routes():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)[0]

    routes = solution.routes()
    assert len(routes) == instance.num_customers
    for route in routes:
        assert len(route) == 1


def test_objective():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)

    objective = solution.objective()
    assert (objective > 0).all()


def test_mean_objective_per_instance_in_batch():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
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
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)
    rollout_size = 5
    expanded_solution = solution.unsqueeze(-1).expand(*solution.batch_size, rollout_size)
    best_per_instance = torch.randint(0, rollout_size, (10,))

    gathered_solution = expanded_solution.gather(1, best_per_instance).squeeze(1)

    assert gathered_solution.batch_dims == 1
    assert gathered_solution.instance.batch_size == (10,)


def test_is_complete():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)

    assert (solution.is_complete == True).all()

    solution.sequence = solution.sequence[:, :-3]

    assert (solution.is_complete == False).all()


def test_deconstructed():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)

    solution.sequence[0, 4] = -1

    assert (solution.is_deconstructed == True)[0]
    assert (solution.is_complete == False)[0]
    assert (solution.is_complete == True)[1:].all()

    solution.sequence[:, 4] = -1

    assert (solution.is_deconstructed == True).all()
    assert (solution.is_complete == False).all()


def test_is_feasible():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)
    assert solution.is_feasible.all()

    infeasible_config = CVRPConfig(
        num_customers=20,
        vehicle_capacity_raw=1,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    infeasible_instance = CVRPSampler(infeasible_config, seed=0)(10)
    infeasible_solution = all_customers_one_route(infeasible_instance)
    assert (~infeasible_solution.is_feasible).all()

    feasible_config = CVRPConfig(
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


@pytest.mark.xfail
def test_is_feasible_on_deconstructed():
    """
    Test if is_feasible is safe to call on deconstructed solutions.
    """

    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
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


def test_route_indexed_sequence():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    num_customers = instance.num_customers

    solution = one_customer_per_route(instance)
    route_indexed_sequence = solution.route_indexed_sequence

    # Expected: [0, 0, 1, 1, 2, 2, ..., 19, 19, 19]
    expected = torch.cat(
        [torch.arange(num_customers).repeat_interleave(2), torch.tensor([num_customers - 1])]
    )
    assert torch.equal(route_indexed_sequence[0], expected)

    solution = all_customers_one_route(instance)
    route_indexed_sequence = solution.route_indexed_sequence

    # Expected: [0, 0, 0, ..., 0]
    expected = torch.zeros_like(route_indexed_sequence[0])
    assert torch.equal(route_indexed_sequence[0], expected)


def test_num_routes():
    instance = CVRPSampler(CVRPConfig.for_n(20))(10)
    solution = one_customer_per_route(instance)
    num_routes = solution.num_routes

    assert (num_routes == 20).all()

    solution = all_customers_one_route(instance)
    num_routes = solution.num_routes

    assert (num_routes == 1).all()


@pytest.mark.xfail
def test_num_routes_ignores_trailing_depot_padding():
    """
    Trailing padding zeros must not affect the semantic number of routes.
    """

    config = CVRPConfig(
        num_customers=3,
        vehicle_capacity_raw=3,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(1)

    # Three non-empty routes: [1], [2], [3], followed by depot padding.
    # Semantic routes should be 3, regardless of how many terminal 0s there are.
    seq = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 0, 0]])
    solution = CVRPSolution(instance=instance, sequence=seq)

    assert solution.num_routes.item() == 3


@pytest.mark.xfail
def test_num_routes_ignores_consecutive_depots_mid_sequence():
    """Consecutive depots should not create empty routes.

    Example: 0-1-0-0-2-0-3-0 encodes the same non-empty routes as 0-1-0-2-0-3-0.
    """

    config = CVRPConfig(
        num_customers=3,
        vehicle_capacity_raw=3,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(1)

    seq = torch.tensor([[0, 1, 0, 0, 2, 0, 3, 0]])
    solution = CVRPSolution(instance=instance, sequence=seq)

    assert solution.num_routes.item() == 3


def test_demand_per_route():
    config = CVRPConfig(
        num_customers=20,
        vehicle_capacity_raw=20,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(10)
    solution = all_customers_one_route(instance)

    assert torch.allclose(solution.demand_per_route(), torch.tensor(1.0))

    solution = one_customer_per_route(instance)
    assert torch.allclose(solution.demand_per_route(), torch.tensor(1.0 / 20))


@pytest.mark.xfail
def test_demand_per_route_on_deconstructed():
    """
    Test if demand_per_route is safe on deconstructed solutions.

    Even if its values are not meaningful for incomplete solutions, it must not crash
    when `self.sequence` contains UNASSIGNED (-1).
    """

    config = CVRPConfig(
        num_customers=20,
        vehicle_capacity_raw=20,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(10)
    solution = one_customer_per_route(instance)

    # Introduce deconstruction tokens
    solution.sequence[0, 4] = -1
    solution.sequence[3, 6] = -1
    assert solution.is_deconstructed[[0, 3]].all()

    # Must not raise
    dpr = solution.demand_per_route()

    assert torch.isfinite(dpr).all()
    assert dpr.shape[0] == 10


def test_route_per_customer():
    """
    Sequence: D -> C1 -> C2 -> D -> C3 -> D

    Routes:
        - Route 0: C0, C1,
        - Route 1: C2
    """
    config = CVRPConfig(
        num_customers=3,
        vehicle_capacity_raw=3,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(1)
    solution = CVRPSolution(instance=instance, sequence=torch.tensor([[0, 1, 2, 0, 3, 0]]))

    route_per_customer = solution.route_per_customer()

    expected = torch.tensor([[0, 0, 1]])
    assert torch.equal(route_per_customer, expected)

    assert torch.equal(route_per_customer, build_encoder_input_indexes(solution)[1])


def test_neighbours_per_customer():
    """
    Sequence: D -> C1 -> C2 -> D -> C3 -> D
    """
    config = CVRPConfig(
        num_customers=3,
        vehicle_capacity_raw=3,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(1)
    solution = CVRPSolution(instance=instance, sequence=torch.tensor([[0, 1, 2, 0, 3, 0]]))

    neighbours_per_customer = solution.neighbours_per_customer()

    expected = torch.tensor([[[0, 2], [1, 0], [0, 0]]])
    assert torch.equal(neighbours_per_customer, expected)

    assert torch.equal(neighbours_per_customer, build_encoder_input_indexes(solution)[0])


def test_remove_customers():
    # Initial sequence: D -> 1 -> 2 -> 3 -> D
    sequence = torch.tensor([[0, 1, 2, 3, 0]])
    config = CVRPConfig(
        num_customers=3,
        vehicle_capacity_raw=3,
        min_demand_raw=1,
        max_demand_raw=1,
    )
    instance = CVRPSampler(config, seed=0)(1)
    solution = CVRPSolution(instance=instance, sequence=sequence, batch_size=[1])

    to_remove = torch.tensor([[[1, 3], [2, 1]]])
    deconstructed_solution = solution.remove_customers(to_remove)

    expected = torch.tensor([[[0, -1, 2, -1, 0], [0, -1, -1, 3, 0]]])
    assert deconstructed_solution.sequence.shape == (1, 2, 5)
    assert torch.equal(deconstructed_solution.sequence, expected)
    assert deconstructed_solution.is_deconstructed.all()
