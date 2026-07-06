import torch

from core.models.embeddings import CVRPInitialEmbedding
from core.models.nds.encoders.config import EncoderConfig
from core.models.nds.encoders.joint import JointEncoder
from core.models.nds.encoders.layers import InstanceNorm
from core.models.nds.encoders.split import SplitNDSEncoder
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.features import (
    CVRPInstanceFeatureExtractor,
    CVRPSolutionFeatureExtractor,
)
from core.problems.cvrp.solve import one_customer_per_route
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.features import (
    TSPInstanceFeatureExtractor,
    TSPSolutionFeatureExtractor,
)
from core.problems.tsp.solve import random_permutation


def _cvrp_encoder_inputs(B: int, N: int):
    sampler = CVRPSampler(CVRP.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)
    instance_features = CVRPInstanceFeatureExtractor()(instance)
    solution_features = CVRPSolutionFeatureExtractor()(solution)
    return instance_features, solution_features


def _tsp_encoder_inputs(B: int, N: int):
    instance = TSPSampler(TSP(num_locations=N), seed=42)(B)
    solution = random_permutation(instance, seed=42)
    instance_features = TSPInstanceFeatureExtractor()(instance)
    solution_features = TSPSolutionFeatureExtractor()(solution)
    return instance_features, solution_features


def test_encoder() -> None:
    B, N = 4, 20
    instance_features, solution_features = _cvrp_encoder_inputs(B, N)

    encoder = JointEncoder(problem_name="cvrp")
    embedded_nodes = encoder.forward(instance_features, solution_features)

    assert embedded_nodes.shape == (B, N + 1, 128)


def test_with_final_norm_resolution() -> None:
    # None follows pre_norm; an explicit value wins.
    assert EncoderConfig(pre_norm=False).with_final_norm is False
    assert EncoderConfig(pre_norm=True).with_final_norm is True
    assert EncoderConfig(pre_norm=True, with_final_norm=False).with_final_norm is False
    assert EncoderConfig(pre_norm=False, with_final_norm=True).with_final_norm is True


def test_default_encoder_has_no_final_norm() -> None:
    encoder = JointEncoder(problem_name="cvrp")
    assert encoder.final_norm is None


def test_pre_norm_encoder_cvrp_forward() -> None:
    B, N = 4, 20
    instance_features, solution_features = _cvrp_encoder_inputs(B, N)

    encoder = JointEncoder(problem_name="cvrp", encoder_config=EncoderConfig(pre_norm=True))
    assert encoder.final_norm is not None

    embedded_nodes = encoder.forward(instance_features, solution_features)
    assert embedded_nodes.shape == (B, N + 1, 128)
    assert not torch.isnan(embedded_nodes).any()


def test_pre_norm_encoder_tsp_forward() -> None:
    # TSP has no depot (offset=0), exercising the all-customers path.
    B, N = 4, 20
    instance_features, solution_features = _tsp_encoder_inputs(B, N)

    encoder = JointEncoder(
        problem_name="tsp",
        encoder_config=EncoderConfig(pre_norm=True),
        with_route_encoding=False,
    )
    embedded_nodes = encoder.forward(instance_features, solution_features)

    assert embedded_nodes.shape == (B, N, 128)
    assert not torch.isnan(embedded_nodes).any()


def test_rms_norm_selected_in_structural_layers() -> None:
    encoder = JointEncoder(problem_name="cvrp", encoder_config=EncoderConfig(norm_type="rms"))
    assert isinstance(encoder.neighbours_encoding.norm, torch.nn.RMSNorm)
    assert isinstance(encoder.route_encoding.norm, torch.nn.RMSNorm)

    default_encoder = JointEncoder(problem_name="cvrp")
    assert isinstance(default_encoder.neighbours_encoding.norm, InstanceNorm)


def test_split_encoder_final_norm_only_at_fusion_output() -> None:
    # Under pre-norm the trailing norm sits at the fusion (decoder-facing) output, not inside
    # the intermediate instance encoder.
    encoder = SplitNDSEncoder(
        problem_name="cvrp",
        encoder_config=EncoderConfig(pre_norm=True),
    )
    assert encoder.instance_encoder.final_norm is None
    assert encoder.fusion.final_norm is not None

    default_encoder = SplitNDSEncoder(problem_name="cvrp")
    assert default_encoder.fusion.final_norm is None


def test_initial_node_embedding() -> None:
    N = 20
    sampler = CVRPSampler(CVRP.for_n(N))
    instance = sampler(4)

    instance_features = CVRPInstanceFeatureExtractor()(instance)
    initial_embedding = CVRPInitialEmbedding(hidden_dim=128)
    initial_node_embeddings = initial_embedding.forward(instance_features)

    assert initial_node_embeddings.shape == (4, N + 1, 128)
