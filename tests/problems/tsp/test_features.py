import pytest
import torch

from core.problems.tsp.features import RepairTSPFeatureExtractor
from core.problems.tsp.instance import TSPInstance
from core.problems.tsp.solution import TSPSolution


@pytest.fixture
def instance_n4():
    locations = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]])
    return TSPInstance.create(locations=locations)


class TestRepairTSPFeatureExtractor:
    @pytest.mark.parametrize(
        "tour, expected_indices, expected_padding",
        [
            (
                [
                    [0, 1, -1, 3]
                ],  # nodes 0 and 1 are connected, 3 is partially, 2 is unassigned. Degrees: 0: 2, 1: 1, 2: 0, 3: 1
                [[1, 2, 3]],
                [[False, False, False]],
            ),
            (
                [[0, 1, 2, 3]],  # all assigned, completely irrelevant
                [[]],
                [[]],
            ),
            (
                [[-1, -1, -1, -1]],  # none assigned
                [[0, 1, 2, 3]],
                [[False, False, False, False]],
            ),
        ],
    )
    def test_extract_subset_single_instance(
        self, instance_n4, tour, expected_indices, expected_padding
    ):
        tour = torch.tensor(tour, dtype=torch.long)
        expected_indices = torch.tensor(expected_indices, dtype=torch.long)
        expected_padding = torch.tensor(expected_padding, dtype=torch.bool)

        expected_k = expected_indices.shape[1]

        solution = TSPSolution.create(instance=instance_n4, tour=tour)
        extractor = RepairTSPFeatureExtractor()
        features = extractor(solution)

        node_features = features["node_features"]
        padding_mask = features["padding_mask"]
        original_indices = features["original_indices"]

        assert node_features.shape == (1, expected_k, 3)
        assert padding_mask.shape == (1, expected_k)
        assert original_indices.shape == (1, expected_k)

        # Verify indices and padding
        if expected_k > 0:
            assert torch.equal(original_indices, expected_indices)
            assert torch.equal(padding_mask, expected_padding)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "tour, expected_indices, expected_padding",
        [
            (
                [
                    [0, 1, -1, 3],  # degrees: [2, 1, 0, 1], free nodes: 1, 2, 3 (count=3)
                    [0, -1, -1, 3],  # degrees: [1, 0, 0, 1], free nodes: 0, 1, 2, 3 (count=4)
                ],
                [
                    [1, 2, 3, 0],  # Node 0 pushed to end (padded here, k=4)
                    [0, 1, 2, 3],
                ],
                [[False, False, False, True], [False, False, False, False]],
            ),
            (
                [
                    [0, 1, 2, 3],  # degrees: [2, 2, 2, 2], free nodes = 0
                    [0, 1, -1, 3],  # degrees: [2, 1, 0, 1], free nodes = 3 (1, 2, 3)
                ],
                [
                    [0, 1, 2],
                    [1, 2, 3],
                ],
                [[True, True, True], [False, False, False]],
            ),
        ],
    )
    def test_extract_subset_batch(self, instance_n4, tour, expected_indices, expected_padding):
        tour = torch.tensor(tour, dtype=torch.long)
        expected_indices = torch.tensor(expected_indices, dtype=torch.long)
        expected_padding = torch.tensor(expected_padding, dtype=torch.bool)

        expected_k = expected_indices.shape[1]
        b_size = tour.shape[0]

        instance = instance_n4.expand(b_size)
        solution = TSPSolution.create(instance=instance, tour=tour)

        extractor = RepairTSPFeatureExtractor()
        features = extractor(solution)

        assert {"node_features", "padding_mask", "original_indices"}.issubset(features.keys())

        node_features = features["node_features"]  # (B, K, 3)
        padding_mask = features["padding_mask"]  # (B, K)
        original_indices = features["original_indices"]  # (B, K)

        assert node_features.shape == (b_size, expected_k, 3)
        assert padding_mask.shape == (b_size, expected_k)
        assert original_indices.shape == (b_size, expected_k)

        if expected_k > 0:
            assert torch.equal(original_indices, expected_indices)
            assert torch.equal(padding_mask, expected_padding)
