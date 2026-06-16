from abc import ABC, abstractmethod


class InstanceFeatureExtractor[I, F](ABC):
    """Maps a problem instance to a typed feature container consumed by a model."""

    @abstractmethod
    def __call__(self, instance: I) -> F: ...


class SolutionFeatureExtractor[S, F](ABC):
    """Maps a solution to a typed feature container consumed by a model."""

    @abstractmethod
    def __call__(self, solution: S) -> F: ...
