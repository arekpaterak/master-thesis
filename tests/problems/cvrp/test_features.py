import pytest
import torch

from core.problems.cvrp.features import RepairCVRPFeatureExtractor
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.solution import CVRPSolution


@pytest.fixture
def instance_n4():
    locations = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.5, 0.5]]])
    demand = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
    capacity = torch.tensor([[3.0]])
    return CVRPInstance.create(
        locations=locations,
        demand=demand,
        capacity=capacity,
    )


class TestRepairCVRPFeatureExtractor:
    @pytest.mark.parametrize(
        "sequence, expected_indices, expected_padding",
        [
            (
                [[0, 1, -1, 4, 0]],
                [[0, 1, 2, 3]],
                [[False, False, False, False]],
            ),
            (
                [[0, 1, 2, 0, 3, 4, 0]],  # all assigned
                [[]],
                [[]],
            ),
            (
                [[0, -1, -1, -1, -1, 0]],  # none assigned
                [[0, 1, 2, 3]],
                [[False, False, False, False]],
            ),
            (
                [[0, 1, 2, 0, -1, 4, 0]],
                [[2, 3]],
                [[False, False]],
            ),
        ],
    )
    def test_extract_subset_single_instance(
        self, instance_n4, sequence, expected_indices, expected_padding
    ):
        sequence = torch.tensor(sequence, dtype=torch.long)
        expected_indices = torch.tensor(expected_indices, dtype=torch.long)
        expected_padding = torch.tensor(expected_padding, dtype=torch.bool)

        expected_k = expected_indices.shape[1]

        solution = CVRPSolution.create(instance=instance_n4, sequence=sequence)
        extractor = RepairCVRPFeatureExtractor()
        features = extractor(solution)

        customer_features = features["customer_features"]
        depot_features = features["depot_features"]
        padding_mask = features["padding_mask"]
        original_indices = features["original_indices"]

        assert customer_features.shape == (1, expected_k, 5)
        assert depot_features.shape == (1, 1, 5)
        assert padding_mask.shape == (1, expected_k)
        assert original_indices.shape == (1, expected_k)

        if expected_k > 0:
            assert torch.equal(original_indices, expected_indices)
            assert torch.equal(padding_mask, expected_padding)

    @pytest.mark.parametrize(
        "sequence, expected_indices, expected_padding",
        [
            (
                [
                    [
                        0,
                        1,
                        2,
                        0,
                        -1,
                        4,
                        0,
                    ],
                    [
                        0,
                        1,
                        -1,
                        4,
                        0,
                        -1,
                        0,
                    ],
                ],
                [
                    [2, 3, 0, 1],
                    [0, 1, 2, 3],
                ],
                [[False, False, True, True], [False, False, False, False]],
            ),
            (
                [
                    [0, 1, 2, 0, 3, 4, 0],
                    [
                        0,
                        1,
                        -1,
                        -1,
                        0,
                        -1,
                        0,
                    ],
                ],
                [[0, 1, 2, 3], [0, 1, 2, 3]],
                [[True, True, True, True], [False, False, False, False]],
            ),
        ],
    )
    def test_extract_subset_batch(self, instance_n4, sequence, expected_indices, expected_padding):
        sequence = torch.tensor(sequence, dtype=torch.long)
        expected_indices = torch.tensor(expected_indices, dtype=torch.long)
        expected_padding = torch.tensor(expected_padding, dtype=torch.bool)

        expected_k = expected_indices.shape[1]
        B = sequence.shape[0]

        instance = instance_n4.expand(B)
        solution = CVRPSolution.create(instance=instance, sequence=sequence)

        extractor = RepairCVRPFeatureExtractor()
        features = extractor(solution)

        assert {"customer_features", "depot_features", "padding_mask", "original_indices"}.issubset(
            features.keys()
        )

        depot_features = features["depot_features"]
        customer_features = features["customer_features"]
        padding_mask = features["padding_mask"]
        original_indices = features["original_indices"]

        assert customer_features.shape == (B, expected_k, 5)
        assert depot_features.shape == (B, 1, 5)
        assert padding_mask.shape == (B, expected_k)
        assert original_indices.shape == (B, expected_k)

        if expected_k > 0:
            assert torch.equal(original_indices, expected_indices)
            assert torch.equal(padding_mask, expected_padding)
