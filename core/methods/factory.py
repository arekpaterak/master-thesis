from __future__ import annotations

from typing import Annotated

import torch
from pydantic import Field

from core.lns.start_solution import with_warmup
from core.methods.base import Method
from core.methods.nds_pretrained import NDSPretrainedConfig, NDSPretrainedMethod
from core.methods.nds_trained import NDSTrainedConfig, NDSTrainedMethod
from core.methods.random_destroy import RandomDestroyMethod, RandomDestroyMethodConfig
from core.problems import ProblemConfig
from core.problems.solve import generate_initial_solution

MethodConfig = Annotated[
    NDSTrainedConfig | NDSPretrainedConfig | RandomDestroyMethodConfig,
    Field(discriminator="type"),
]


def build_method(
    config: NDSTrainedConfig | NDSPretrainedConfig | RandomDestroyMethodConfig,
    *,
    problem: ProblemConfig,
    device: torch.device,
) -> Method:
    """Build a Method from any method config, dispatching on its concrete type."""
    start_solution = with_warmup(
        generate_initial_solution, problem_name=problem.name, search=config.search
    )
    if isinstance(config, NDSTrainedConfig):
        return NDSTrainedMethod.build(config, generate_start_solution=start_solution, device=device)
    if isinstance(config, NDSPretrainedConfig):
        return NDSPretrainedMethod.build(
            config,
            problem=problem,
            generate_start_solution=start_solution,
            device=device,
        )
    if isinstance(config, RandomDestroyMethodConfig):
        return RandomDestroyMethod.build(
            config, problem=problem, generate_start_solution=start_solution
        )
    raise TypeError(f"Unknown method config: {type(config).__name__}")
