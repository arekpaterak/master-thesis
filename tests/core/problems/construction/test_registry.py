import pytest

from core.problems.cmstp.construction import EsauWilliamsConstruction, StarConstruction
from core.problems.construction import ConstructionHeuristic


class _OtherProblemOnly(ConstructionHeuristic):
    construction_name = "other_problem_only"
    problem_name = "not-cmstp"

    def __call__(self, instance):
        raise NotImplementedError


def test_create_resolves_registered_cmstp_heuristics():
    assert isinstance(ConstructionHeuristic.create("cmstp", "star"), StarConstruction)
    assert isinstance(
        ConstructionHeuristic.create("cmstp", "esau_williams"), EsauWilliamsConstruction
    )


def test_create_unknown_type_raises():
    with pytest.raises(ValueError, match="Unsupported construction name"):
        ConstructionHeuristic.create("cmstp", "does_not_exist")


def test_create_unimplemented_for_problem_raises():
    with pytest.raises(ValueError, match="not implemented for problem 'cmstp'"):
        ConstructionHeuristic.create("cmstp", "other_problem_only")
