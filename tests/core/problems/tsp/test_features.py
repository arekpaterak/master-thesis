import pytest
import torch

from core.problems.tsp.features import (
    TSPInstanceFeatureExtractor,
    TSPSolutionFeatureExtractor,
)
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.instance import TSPInstance
from core.problems.tsp.solution import TSPSolution


@pytest.fixture
def instance_n4():
    locations = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]])
    return TSPInstance.create(locations=locations)


@pytest.fixture
def graph_solution(instance_n4):
    return TSPGraphSolution.create_from_sequence(instance_n4, torch.tensor([[0, 1, 2, 3]]))


def test_instance_features_shapes(instance_n4):
    features = TSPInstanceFeatureExtractor()(instance_n4)
    assert features.node_locations.shape == (1, 4, 2)
    assert features.node_features.shape == (1, 4, 2)
    assert features.locations.shape == (1, 4, 2)


def test_solution_features_off(graph_solution):
    features = TSPSolutionFeatureExtractor()(graph_solution)
    assert features.neighbours_index.shape == (1, 4, 2)
    assert features.tour_phase is None


def test_solution_features_tour_phase_when_positional(graph_solution):
    features = TSPSolutionFeatureExtractor(with_positional=True)(graph_solution)
    assert features.tour_phase is not None
    assert features.tour_phase.shape == (1, 4)


def test_solution_features_positional_on_sequence_raises(instance_n4):
    solution = TSPSolution.create(instance=instance_n4, tour=torch.tensor([[0, 1, 2, 3]]))
    with pytest.raises(NotImplementedError):
        TSPSolutionFeatureExtractor(with_positional=True)(solution)
