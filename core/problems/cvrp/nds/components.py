from core.models.nds.components import NDSComponents
from core.models.nds.encoders.layers import NeighboursEncodingLayer
from core.problems.cvrp.embedding import CVRPInitialEmbedding
from core.problems.cvrp.features import CVRPInstanceFeatureExtractor, CVRPSolutionFeatureExtractor
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.problem import PROBLEM


def _feature_extractors(
    with_positional: bool,
) -> tuple[CVRPInstanceFeatureExtractor, CVRPSolutionFeatureExtractor]:
    return CVRPInstanceFeatureExtractor(
        with_positional=with_positional
    ), CVRPSolutionFeatureExtractor(with_positional=with_positional)


NDS = NDSComponents[CVRPInstance, CVRPGraphSolution](
    problem=PROBLEM,
    num_context_nodes=CVRPInstance.num_fixed_nodes,
    feature_extractors=_feature_extractors,
    initial_embedding=CVRPInitialEmbedding,
    neighbours_layer=NeighboursEncodingLayer,
    route_encoding=True,
    cross_route_positional=True,
)
