import torch

from core.lns.operators.repair.base import RepairOperator
from core.models.attention_model import AttentionModel
from core.problems.tsp.solution import TSPSolution
from core.random import SeededGenerator


# Legacy
class AttentionModelRepair(RepairOperator[TSPSolution]):
    def __init__(self, model: AttentionModel, greedy: bool = True, seed: int | None = None):
        self.model = model
        self.greedy = greedy
        self._rng = SeededGenerator(seed)

    def __call__(self, deconstructed_solution: TSPSolution) -> TSPSolution:
        partial_tour = self._extract_valid_tour(deconstructed_solution.tour)

        rollout = self.model.rollout(
            deconstructed_solution.instance,
            greedy=self.greedy,
            previous_actions=partial_tour,
            rng=self._rng,
        )

        return TSPSolution.create(
            instance=deconstructed_solution.instance,
            tour=rollout.actions,
        )

    @staticmethod
    def _extract_valid_tour(tour: torch.Tensor) -> torch.Tensor:
        shifts = -(find_first_following_value(tour, value=-1))
        shifted = batched_roll_2d(tour, shifts=shifts)

        ks = (shifted == -1).sum(dim=1)
        k = ks[0]
        assert (ks == k).all(), "All batches must have the same number of removed nodes."

        valid_tour = shifted[:, :-k]  # (batch, N-k)

        assert (valid_tour != -1).all(), "Valid tour should not contain any -1 values."

        return valid_tour


def batched_roll_2d(x: torch.Tensor, *, shifts: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "Expected a 2D tensor of shape (batch_size, sequence_length)."
    assert shifts.ndim == 1, "Expected shifts to be a 1D tensor of shape (batch_size,)."
    assert x.shape[0] == shifts.shape[0], "Batch size of x and shifts must match."

    B, L = x.shape
    idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    shifted_idx = (idx - shifts.unsqueeze(1)) % L
    return torch.gather(x, 1, shifted_idx)


def find_first_following_value(x: torch.Tensor, *, value: float = -1):
    assert x.ndim == 2, "Expected a 2D tensor of shape (batch_size, sequence_length)."

    is_value = x == value
    is_not_value = x != value

    # We need (x[i] != -1) AND (x[i-1] == -1)
    # Shift 'is_value' to the right by 1 to align with 'is_not_value'
    # We pad the left with False because index 0 has no "previous" element
    was_value = torch.cat(
        [torch.zeros((x.size(0), 1), dtype=torch.bool, device=x.device), is_value[:, :-1]], dim=1
    )
    transition_mask = was_value & is_not_value

    return torch.argmax(transition_mask.long(), dim=1)
