from core.models.nds.components import NDSComponents
from core.models.nds.encoders.layers import NeighboursEncodingLayer
from core.problems.tsp.embedding import TSPInitialEmbedding
from core.problems.tsp.features import TSPInstanceFeatureExtractor, TSPSolutionFeatureExtractor
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.instance import TSPInstance
from core.problems.tsp.problem import PROBLEM


def _feature_extractors(
    with_positional: bool,
) -> tuple[TSPInstanceFeatureExtractor, TSPSolutionFeatureExtractor]:
    return TSPInstanceFeatureExtractor(), TSPSolutionFeatureExtractor(
        with_positional=with_positional
    )


NDS = NDSComponents[TSPInstance, TSPGraphSolution](
    problem=PROBLEM,
    num_context_nodes=TSPInstance.num_fixed_nodes,
    feature_extractors=_feature_extractors,
    initial_embedding=TSPInitialEmbedding,
    neighbours_layer=NeighboursEncodingLayer,
    route_encoding=False,
    cross_route_positional=False,
)
