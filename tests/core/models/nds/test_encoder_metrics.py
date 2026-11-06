import math

import torch

from core.models.nds.config import JointModelConfig, SplitModelConfig
from core.models.nds.decoders.config import SimpleOnePassDecoderConfig
from core.models.nds.metrics import EncoderMetricsRecorder, mean_pairwise_cosine_similarity
from core.models.nds.models import NDSModel
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.registry import build_nds_model


def test_orthogonal_embeddings_have_zero_cosine() -> None:
    embeddings = torch.eye(3).unsqueeze(0)  # (1, N=3, D=3)

    assert mean_pairwise_cosine_similarity(embeddings) == 0.0


def test_identical_embeddings_have_full_cosine() -> None:
    embeddings = torch.ones(1, 3, 4)  # (1, N=3, D=4) - the collapse case

    assert math.isclose(mean_pairwise_cosine_similarity(embeddings), 1.0, rel_tol=1e-6)


def test_antipodal_embeddings_have_negative_cosine() -> None:
    embeddings = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])  # (1, N=2, D=2)

    assert math.isclose(mean_pairwise_cosine_similarity(embeddings), -1.0, rel_tol=1e-6)


def test_single_node_is_degenerate() -> None:
    embeddings = torch.ones(1, 1, 4)  # (1, N=1, D=4)

    assert mean_pairwise_cosine_similarity(embeddings) == 0.0


def test_mixed_case_matches_hand_computation() -> None:
    # (1, N=3, D=2): two collinear embeddings and one orthogonal to both.
    embeddings = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    # Distinct unordered pairs: (0,1) -> 1, (0,2) -> 0, (1,2) -> 0; mean over 6 ordered = 2/6.

    assert math.isclose(mean_pairwise_cosine_similarity(embeddings), 1 / 3, rel_tol=1e-6)


def test_cosine_is_invariant_to_embedding_magnitude() -> None:
    # Case above with the first row scaled: cosine ignores magnitude, so the mean is unchanged.
    embeddings = torch.tensor([[[10.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

    assert math.isclose(mean_pairwise_cosine_similarity(embeddings), 1 / 3, rel_tol=1e-6)


def test_zero_embedding_contributes_zero_not_nan() -> None:
    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])  # (1, N=2, D=2)

    result = mean_pairwise_cosine_similarity(embeddings)
    assert result == 0.0
    assert not math.isnan(result)


def test_leading_batch_dims_reduce_to_scalar() -> None:
    # (B=2, A=2, N=2, D=2): one orthogonal instance (0.0) and one identical instance (1.0).
    orthogonal = torch.eye(2)
    identical = torch.ones(2, 2)
    embeddings = torch.stack([orthogonal, orthogonal, identical, identical]).reshape(2, 2, 2, 2)

    assert math.isclose(mean_pairwise_cosine_similarity(embeddings), 0.5, rel_tol=1e-6)


def test_half_precision_input_matches_float_result() -> None:
    embeddings = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float16)

    assert math.isclose(mean_pairwise_cosine_similarity(embeddings), 1 / 3, rel_tol=1e-3)


def test_recorder_averages_over_recorded_steps() -> None:
    recorder = EncoderMetricsRecorder()
    assert math.isnan(recorder.embeddings_cos_sim)

    recorder.record(torch.eye(3).unsqueeze(0))  # 0.0
    recorder.record(torch.ones(1, 3, 4))  # 1.0
    assert math.isclose(recorder.embeddings_cos_sim, 0.5, rel_tol=1e-6)

    recorder.reset()
    assert math.isnan(recorder.embeddings_cos_sim)


def test_recorder_offset_drops_leading_nodes() -> None:
    # A collapsed depot at index 0 must not mask an otherwise orthogonal customer set.
    embeddings = torch.tensor(
        [[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )

    with_depot = EncoderMetricsRecorder(offset=0)
    without_depot = EncoderMetricsRecorder(offset=1)
    with_depot.record(embeddings)
    without_depot.record(embeddings)

    assert without_depot.embeddings_cos_sim == 0.0
    assert with_depot.embeddings_cos_sim > 0.0


def test_recorder_does_not_track_gradients() -> None:
    embeddings = torch.ones(1, 3, 4, requires_grad=True)
    recorder = EncoderMetricsRecorder()

    recorder.record(embeddings)

    assert math.isclose(recorder.embeddings_cos_sim, 1.0, rel_tol=1e-6)
    assert embeddings.grad is None


def _cvrp_solution(B: int = 4, N: int = 20) -> CVRPGraphSolution:
    instance = CVRPSampler(CVRPDistribution.for_n(N))(B)
    return CVRPGraphSolution.one_customer_per_route(instance)


def _split_model():
    return build_nds_model(
        SplitModelConfig(problem_name="cvrp", decoder=SimpleOnePassDecoderConfig())
    )


def _joint_model():
    return build_nds_model(
        JointModelConfig(problem_name="cvrp", decoder=SimpleOnePassDecoderConfig())
    )


def _attach(model: NDSModel):
    return EncoderMetricsRecorder.attach(model.encoder, offset=model.components.num_context_nodes)


def test_attach_records_once_per_rollout_for_both_models() -> None:
    # The hook must fire for both encoders through model.encoder. Split's forward is the renamed
    # fuse, so this guards against that path silently going unmeasured.
    solution = _cvrp_solution()
    for model in (_split_model().eval(), _joint_model().eval()):
        with _attach(model) as recorder:
            model.rollout(solution, rollouts_per_solution=3, deconstruction_size=5)
            model.rollout(solution, rollouts_per_solution=3, deconstruction_size=5)

        assert recorder._embeddings_cos_sim._total_weight == 2
        assert -1.0 <= recorder.embeddings_cos_sim <= 1.0


def test_attach_detaches_after_the_block() -> None:
    model = _split_model().eval()
    solution = _cvrp_solution()

    with _attach(model) as recorder:
        model.rollout(solution, rollouts_per_solution=3, deconstruction_size=5)
    count_after_block = recorder._embeddings_cos_sim._total_weight

    model.rollout(solution, rollouts_per_solution=3, deconstruction_size=5)
    assert recorder._embeddings_cos_sim._total_weight == count_after_block == 1


def test_attach_excludes_the_depot() -> None:
    # The recorder built for a CVRP model must slice off the depot (offset 1).
    model = _split_model()
    with _attach(model) as recorder:
        assert recorder._offset == model.components.num_context_nodes == 1
