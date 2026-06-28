import math

import pytest
import torch

from core.models.embeddings.positional import circular_sinusoidal_encoding
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.solution import CVRPSolution
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.instance import TSPInstance
from core.problems.tsp.solution import TSPSolution


def _cvrp_instance() -> CVRPInstance:
    # Depot at origin; customers on the axes at unit and double-unit distances.
    locations = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]])
    demand = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
    capacity = torch.tensor([[10.0]])
    return CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)


def test_cvrp_route_phase_matches_hand_computation():
    instance = _cvrp_instance()
    # Two routes: depot->1->2->depot and depot->3->4->depot. Each route length is 4.
    sequence = torch.tensor([[0, 1, 2, 0, 3, 4, 0]])
    solution = CVRPGraphSolution.create_from_sequence(instance, sequence)

    phase = solution.route_phase()  # (1, 4)
    expected = torch.tensor([[math.pi / 2, math.pi, math.pi / 2, math.pi]])
    torch.testing.assert_close(phase, expected)


def test_cvrp_route_phase_single_customer_route_is_pi():
    instance = _cvrp_instance()
    solution = CVRPGraphSolution.one_customer_per_route(instance)
    phase = solution.route_phase()
    torch.testing.assert_close(phase, torch.full((1, 4), math.pi))


def test_cvrp_sequence_route_phase_not_implemented():
    instance = _cvrp_instance()
    solution = CVRPSolution.create(
        instance=instance, sequence=torch.tensor([[0, 1, 2, 0, 3, 4, 0]])
    )
    with pytest.raises(NotImplementedError):
        solution.route_phase()


def test_cvrp_depot_polar_angle():
    instance = _cvrp_instance()
    angle = instance.depot_anchored_polar_angle()  # (1, 4) in [-pi, pi]
    expected = torch.tensor([[0.0, 0.0, math.pi / 2, math.pi / 2]])
    torch.testing.assert_close(angle, expected)


def test_tsp_tour_phase_matches_hand_computation():
    # Unit square traversed 0->1->2->3->0; total length 4.
    locations = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    instance = TSPInstance.create(locations=locations)
    tour = torch.tensor([[0, 1, 2, 3]])
    solution = TSPGraphSolution.create_from_sequence(instance, tour)

    phase = solution.tour_phase()  # (1, 4)
    expected = torch.tensor([[0.0, math.pi / 2, math.pi, 3 * math.pi / 2]])
    torch.testing.assert_close(phase, expected)


def test_tsp_sequence_tour_phase_not_implemented():
    locations = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    instance = TSPInstance.create(locations=locations)
    solution = TSPSolution.create(instance=instance, tour=torch.tensor([[0, 1, 2, 3]]))
    with pytest.raises(NotImplementedError):
        solution.tour_phase()


def test_circular_sinusoidal_encoding_integer_schedule_is_reversal_invariant():
    phase = torch.rand(3, 5) * (2 * math.pi)

    encoding = circular_sinusoidal_encoding(phase, dim=8, direction_aware=False)
    assert encoding.shape == (3, 5, 8)

    # Integer harmonics: cos(k*(2*pi - x)) == cos(k*x), so the direction-invariant
    # variant is exactly reversal-symmetric (the default schedule here).
    reversed_encoding = circular_sinusoidal_encoding(
        (2 * math.pi - phase) % (2 * math.pi), dim=8, direction_aware=False
    )
    torch.testing.assert_close(encoding, reversed_encoding)


def test_circular_sinusoidal_encoding_geometric_schedule_not_reversal_invariant():
    phase = torch.rand(3, 5) * (2 * math.pi)
    encoding = circular_sinusoidal_encoding(
        phase, dim=8, direction_aware=False, schedule="geometric"
    )
    reversed_encoding = circular_sinusoidal_encoding(
        (2 * math.pi - phase) % (2 * math.pi), dim=8, direction_aware=False, schedule="geometric"
    )
    assert not torch.allclose(encoding, reversed_encoding)


def test_circular_sinusoidal_encoding_direction_aware_distinguishes_reversal():
    phase = torch.tensor([1.0, 2.0, 3.0])
    encoding = circular_sinusoidal_encoding(phase, dim=8, direction_aware=True)
    reversed_encoding = circular_sinusoidal_encoding(
        (2 * math.pi - phase) % (2 * math.pi), dim=8, direction_aware=True
    )
    assert encoding.shape == (3, 8)
    assert not torch.allclose(encoding, reversed_encoding)
