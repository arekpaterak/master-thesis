from __future__ import annotations

from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def update(self, value: float, weight: float = 1.0) -> None: ...

    @abstractmethod
    def compute(self) -> float: ...

    @abstractmethod
    def reset(self) -> None: ...


class MeanMetric(Metric):
    def __init__(self) -> None:
        self.reset()

    def update(self, value: float, weight: float = 1.0) -> None:
        self._weighted_sum += value * weight
        self._total_weight += weight

    def compute(self) -> float:
        if self._total_weight == 0.0:
            return float("nan")
        return self._weighted_sum / self._total_weight

    def reset(self) -> None:
        self._weighted_sum = 0.0
        self._total_weight = 0.0


class SumMetric(Metric):
    """Running (optionally weighted) sum of the updated values.

    Unlike MeanMetric an empty sum is 0.0, the additive identity, rather than nan.
    """

    def __init__(self) -> None:
        self.reset()

    def update(self, value: float, weight: float = 1.0) -> None:
        self._sum += value * weight

    def compute(self) -> float:
        return self._sum

    def reset(self) -> None:
        self._sum = 0.0


class MetricCollection:
    def __init__(self, metrics: dict[str, Metric]) -> None:
        self._metrics = metrics

    def update(self, values: dict[str, float], weight: float = 1.0) -> None:
        if values.keys() != self._metrics.keys():
            raise KeyError(f"Expected values for {sorted(self._metrics)}, got {sorted(values)}")
        for name, value in values.items():
            self._metrics[name].update(value, weight)

    def compute(self) -> dict[str, float]:
        return {name: metric.compute() for name, metric in self._metrics.items()}

    def reset(self) -> None:
        for metric in self._metrics.values():
            metric.reset()
