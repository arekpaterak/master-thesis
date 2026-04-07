import pytest
import torch

from core.lns.operators.repair import AttentionModelRepair, batched_roll_2d


@pytest.mark.parametrize(
    "tours, expected",
    [
        (
            torch.tensor(
                [
                    [-1, 1, 2, -1, -1],
                    [-1, -1, 7, 8, -1],
                    [1, -1, -1, -1, 3],
                    [1, 2, -1, -1, -1],
                ]
            ),
            torch.tensor(
                [
                    [1, 2],
                    [7, 8],
                    [3, 1],
                    [1, 2],
                ]
            ),
        ),
    ],
)
def test_repair_extract_valid_tour(tours, expected):
    valid_tours = AttentionModelRepair._extract_valid_tour(tours)
    torch.testing.assert_close(valid_tours, expected)


@pytest.mark.parametrize(
    "tours, shifts, expected",
    [
        (
            torch.tensor(
                [
                    [-1, 1, 2, 3, 4],
                    [-1, -1, 7, 8, -1],
                    [1, 2, -1, -1, 3],
                    [1, 2, 3, 4, -1],
                ]
            ),
            torch.tensor([-1, -2, 1, 0]),
            torch.tensor(
                [
                    [1, 2, 3, 4, -1],
                    [7, 8, -1, -1, -1],
                    [3, 1, 2, -1, -1],
                    [1, 2, 3, 4, -1],
                ]
            ),
        ),
    ],
)
def test_batched_roll_2d(tours, shifts, expected):
    rolled = batched_roll_2d(tours, shifts=shifts)
    torch.testing.assert_close(rolled, expected)
