from typing import get_args

import pytest

from core.lns.apply_and_repair import build_apply_and_repair
from core.lns.config import SearchParams
from core.problems import registry
from core.problems.cmstp.construction import esau_williams, star
from core.problems.names import PROBLEM_NAMES
from core.problems.problem_config import ProblemConfig


def test_every_config_has_a_problem_and_lns_components() -> None:
    config_names = {
        cls.model_fields["name"].default for cls in get_args(get_args(ProblemConfig)[0])
    }
    assert config_names == set(PROBLEM_NAMES) == set(registry.PROBLEMS) == set(registry.LNS)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_components_match_their_problem(name: str) -> None:
    lns = registry.lns(name)
    assert lns.problem is registry.problem(name)
    assert lns.problem.name == name
    assert "greedy_insert" in lns.apply_and_repairs


@pytest.mark.parametrize("name", ["tsp", "cvrp"])
def test_routing_problems_have_torch_repairs(name: str) -> None:
    assert {"greedy_insert", "greedy_reconnect"} <= set(registry.lns(name).repairs)


def test_cmstp_torch_backend_names_the_missing_repair() -> None:
    with pytest.raises(
        ValueError, match=r"No repair 'greedy_insert' for problem 'cmstp'\. Available: none\."
    ):
        build_apply_and_repair(SearchParams(), lns=registry.lns("cmstp"), backend="torch")


def test_unknown_problem_lists_available() -> None:
    with pytest.raises(ValueError, match=r"No problem 'vrptw'\. Available: cmstp, cvrp, tsp\."):
        registry.problem("vrptw")


def test_cmstp_constructions() -> None:
    cmstp = registry.problem("cmstp")
    assert cmstp.construction("star") is star
    assert cmstp.construction("esau_williams") is esau_williams


def test_unknown_construction_names_the_problem() -> None:
    with pytest.raises(
        ValueError, match=r"No construction 'star' for problem 'cvrp'\. Available: none\."
    ):
        registry.problem("cvrp").construction("star")
