from core.models.nds.components import NDSComponents
from core.models.nds.encoders.layers import TreeNeighboursEncodingLayer
from core.problems.cmstp.embedding import CMSTPInitialEmbedding
from core.problems.cmstp.features import (
    CMSTPInstanceFeatureExtractor,
    CMSTPSolutionFeatureExtractor,
)
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.cmstp.problem import PROBLEM
from core.problems.cmstp.solution import CMSTPSolution


def _feature_extractors(
    with_positional: bool,
) -> tuple[CMSTPInstanceFeatureExtractor, CMSTPSolutionFeatureExtractor]:
    if with_positional:
        raise NotImplementedError("CMSTP positional encoding is not implemented yet.")
    return CMSTPInstanceFeatureExtractor(), CMSTPSolutionFeatureExtractor()


NDS = NDSComponents[CMSTPInstance, CMSTPSolution](
    problem=PROBLEM,
    num_context_nodes=CMSTPInstance.num_fixed_nodes,
    feature_extractors=_feature_extractors,
    initial_embedding=CMSTPInitialEmbedding,
    neighbours_layer=TreeNeighboursEncodingLayer,
    route_encoding=True,
    cross_route_positional=False,
)
