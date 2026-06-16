import pytest
import torch

from core.problems.cvrp.features import (
    CVRPInstanceFeatureExtractor,
    CVRPSolutionFeatureExtractor,
)
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.solution import CVRPSolution


@pytest.fixture
def instance_n4():
    locations = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.5, 0.5]]])
    demand = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
    capacity = torch.tensor([[3.0]])
    return CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)


@pytest.fixture
def graph_solution(instance_n4):
    return CVRPGraphSolution.create_from_sequence(
        instance_n4, torch.tensor([[0, 1, 2, 0, 3, 4, 0]])
    )


def test_instance_features_shapes_and_composed_properties(instance_n4):
    features = CVRPInstanceFeatureExtractor()(instance_n4)
    assert features.depot_location.shape == (1, 1, 2)
    assert features.customer_locations.shape == (1, 4, 2)
    assert features.customer_demand.shape == (1, 4, 1)
    assert features.customer_features.shape == (1, 4, 3)
    assert features.depot_features.shape == (1, 1, 2)
    assert features.locations.shape == (1, 5, 2)
    assert features.depot_angle is None


def test_instance_features_depot_angle_when_positional(instance_n4):
    features = CVRPInstanceFeatureExtractor(with_positional=True)(instance_n4)
    assert features.depot_angle is not None
    assert features.depot_angle.shape == (1, 4)


def test_solution_features_off(graph_solution):
    features = CVRPSolutionFeatureExtractor()(graph_solution)
    assert features.neighbours_index.shape == (1, 4, 2)
    assert features.route_index.shape == (1, 4)
    assert features.route_phase is None


def test_solution_features_route_phase_when_positional(graph_solution):
    features = CVRPSolutionFeatureExtractor(with_positional=True)(graph_solution)
    assert features.route_phase is not None
    assert features.route_phase.shape == (1, 4)


def test_solution_features_positional_on_sequence_raises(instance_n4):
    solution = CVRPSolution.create(
        instance=instance_n4, sequence=torch.tensor([[0, 1, 2, 0, 3, 4, 0]])
    )
    with pytest.raises(NotImplementedError):
        CVRPSolutionFeatureExtractor(with_positional=True)(solution)


def test_solution_features_off_works_on_sequence(instance_n4):
    solution = CVRPSolution.create(
        instance=instance_n4, sequence=torch.tensor([[0, 1, 2, 0, 3, 4, 0]])
    )
    features = CVRPSolutionFeatureExtractor()(solution)
    assert features.neighbours_index.shape == (1, 4, 2)
    assert features.route_phase is None
