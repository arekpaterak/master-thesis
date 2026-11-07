from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from core.models.nds.encoders.layers import NeighboursEncodingLayer, TreeNeighboursEncodingLayer
from core.problems.base import Problem
from core.problems.features import InstanceFeatureExtractor, SolutionFeatureExtractor
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


@dataclass(frozen=True)
class NDSComponents[I: BaseProblemInstance, S: BaseSolution]:
    problem: Problem[I, S]
    num_context_nodes: int
    """Leading nodes the decoder treats as context, never as candidates (e.g. the depot)."""
    feature_extractors: Callable[
        [bool], tuple[InstanceFeatureExtractor[I, object], SolutionFeatureExtractor[S, object]]
    ]
    """Instance and solution feature extractors, with or without positional features."""
    initial_embedding: type[nn.Module]
    neighbours_layer: type[NeighboursEncodingLayer | TreeNeighboursEncodingLayer]
    route_encoding: bool
    """Whether solutions have routes to encode; the default for ``with_route_encoding``."""
    cross_route_positional: bool
    """Whether the positional encoding has a cross-route component next to the in-route one."""
