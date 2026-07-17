import math

import torch

from core.models.nds.metrics import (
    RolloutMetricsRecorder,
    diversity_overlap,
    unique_rollouts,
)


def test_identical_rollouts_have_full_overlap_and_minimal_uniqueness():
    actions = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]).unsqueeze(0)  # (1, R=3, M=3)

    assert diversity_overlap(actions) == 1.0
    assert math.isclose(unique_rollouts(actions), 1 / 3, rel_tol=1e-6)


def test_disjoint_rollouts_have_no_overlap_and_full_uniqueness():
    actions = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).unsqueeze(0)  # (1, R=3, M=3)

    assert diversity_overlap(actions) == 0.0
    assert unique_rollouts(actions) == 1.0


def test_single_rollout_is_degenerate():
    actions = torch.tensor([[0, 1, 2]]).unsqueeze(0)  # (1, R=1, M=3)

    assert diversity_overlap(actions) == 0.0
    assert unique_rollouts(actions) == 1.0


def test_mixed_case_matches_hand_computation():
    # R=3 rollouts. Nodes and the rollouts that pick them:
    #   0 -> {r0, r1}, 1 -> {r0, r1}, 2 -> {r0}, 3 -> {r1}, 4 -> {r2}, 5 -> {r2}, 6 -> {r2}
    actions = torch.tensor([[0, 1, 2], [0, 1, 3], [4, 5, 6]]).unsqueeze(0)

    # Per membership, fraction of the other R-1=2 rollouts that share the node:
    #   r0: 1/2, 1/2, 0 ; r1: 1/2, 1/2, 0 ; r2: 0, 0, 0  -> mean over 9 picks = 2/9
    assert math.isclose(diversity_overlap(actions), 2 / 9, rel_tol=1e-6)
    # r2 duplicates nobody; r0 and r1 differ -> all three sets distinct.
    assert unique_rollouts(actions) == 1.0


def test_duplicate_set_reduces_uniqueness():
    # r1 duplicates r0 (order ignored); r2 is distinct -> 2 distinct sets of 3.
    actions = torch.tensor([[0, 1, 2], [2, 1, 0], [3, 4, 5]]).unsqueeze(0)

    assert math.isclose(unique_rollouts(actions), 2 / 3, rel_tol=1e-6)


def test_leading_batch_dims_reduce_to_scalar():
    # (B=2, A=2, R=2, M=2): first instance identical rollouts, others disjoint.
    identical = torch.tensor([[0, 1], [0, 1]])
    disjoint = torch.tensor([[0, 1], [2, 3]])
    actions = torch.stack([torch.stack([identical, disjoint]), torch.stack([disjoint, disjoint])])
    assert actions.shape == (2, 2, 2, 2)

    # Overlap is the mean over 4 instances: one has overlap 1.0, three have 0.0.
    assert math.isclose(diversity_overlap(actions), 0.25, rel_tol=1e-6)
    # Uniqueness mean: identical -> 0.5, disjoint -> 1.0 ; mean over 4 -> 0.875.
    assert math.isclose(unique_rollouts(actions), 0.875, rel_tol=1e-6)


def test_recorder_averages_over_recorded_steps():
    recorder = RolloutMetricsRecorder()
    assert recorder.diversity_overlap == 0.0
    assert recorder.unique_rollouts == 0.0

    identical = torch.tensor([[0, 1, 2], [0, 1, 2]]).unsqueeze(0)  # overlap 1.0, unique 0.5
    disjoint = torch.tensor([[0, 1, 2], [3, 4, 5]]).unsqueeze(0)  # overlap 0.0, unique 1.0
    recorder.record(identical)
    recorder.record(disjoint)

    assert math.isclose(recorder.diversity_overlap, 0.5, rel_tol=1e-6)
    assert math.isclose(recorder.unique_rollouts, 0.75, rel_tol=1e-6)

    recorder.reset()
    assert recorder.diversity_overlap == 0.0
    assert recorder.unique_rollouts == 0.0
