from collections.abc import Mapping


def resolve[K, V](choices: Mapping[K, V], key: K, *, kind: str, problem: str | None = None) -> V:
    try:
        return choices[key]
    except KeyError:
        scope = f" for problem {problem!r}" if problem is not None else ""
        available = ", ".join(sorted(map(str, choices))) or "none"
        raise ValueError(f"No {kind} {key!r}{scope}. Available: {available}.") from None
