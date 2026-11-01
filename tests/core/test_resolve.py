import pytest

from core.resolve import resolve


def test_returns_value_for_known_key() -> None:
    assert resolve({"a": 1, "b": 2}, "b", kind="thing") == 2


def test_unknown_key_lists_available_choices() -> None:
    with pytest.raises(ValueError, match=r"No repair 'x' for problem 'tsp'\. Available: a, b\."):
        resolve({"b": 2, "a": 1}, "x", kind="repair", problem="tsp")


def test_empty_choices() -> None:
    with pytest.raises(ValueError, match=r"No repair 'x'\. Available: none\."):
        resolve({}, "x", kind="repair")
