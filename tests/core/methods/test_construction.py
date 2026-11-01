import pytest
import torch

from core.methods import ConstructionMethodConfig
from core.problems import make_problem_config
from core.problems.cmstp.construction import esau_williams

CPU = torch.device("cpu")


def test_builds_the_named_construction_without_search() -> None:
    method = ConstructionMethodConfig(construction_name="esau_williams").build(
        problem=make_problem_config(name="cmstp", size=20), device=CPU
    )
    assert method.generate_start_solution is esau_williams
    assert method.search.num_improvement_steps == 0


def test_unknown_construction_names_the_problem() -> None:
    with pytest.raises(ValueError, match=r"No construction 'star' for problem 'cvrp'"):
        ConstructionMethodConfig(construction_name="star").build(
            problem=make_problem_config(name="cvrp", size=20), device=CPU
        )
