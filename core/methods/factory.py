from __future__ import annotations

from typing import Annotated

import torch
from pydantic import Field

from core.lns.start_solution import with_warmup
from core.methods.base import Method
from core.methods.construction import ConstructionMethod, ConstructionMethodConfig
from core.methods.nds_pretrained import NDSPretrainedConfig, NDSPretrainedMethod
from core.methods.nds_trained import NDSTrainedConfig, NDSTrainedMethod
from core.methods.random_destroy import RandomDestroyMethod, RandomDestroyMethodConfig
from core.problems import ProblemConfig, registry

MethodConfig = Annotated[
    NDSTrainedConfig | NDSPretrainedConfig | RandomDestroyMethodConfig | ConstructionMethodConfig,
    Field(discriminator="type"),
]


def build_method(
    config: NDSTrainedConfig
    | NDSPretrainedConfig
    | RandomDestroyMethodConfig
    | ConstructionMethodConfig,
    *,
    problem: ProblemConfig,
    device: torch.device,
) -> Method:
    """Build a Method from any method config, dispatching on its concrete type."""
    # Construction methods build their own start solution and run no improvement search.
    if isinstance(config, ConstructionMethodConfig):
        return ConstructionMethod.build(config, problem=registry.problem(problem.name))

    lns = registry.lns(problem.name)
    start_solution = with_warmup(lns.problem.initial_solution, lns=lns, search=config.search)
    if isinstance(config, NDSTrainedConfig):
        return NDSTrainedMethod.build(
            config, lns=lns, generate_start_solution=start_solution, device=device
        )
    if isinstance(config, NDSPretrainedConfig):
        return NDSPretrainedMethod.build(
            config, lns=lns, generate_start_solution=start_solution, device=device
        )
    if isinstance(config, RandomDestroyMethodConfig):
        return RandomDestroyMethod.build(config, lns=lns, generate_start_solution=start_solution)
    raise TypeError(f"Unknown method config: {type(config).__name__}")
