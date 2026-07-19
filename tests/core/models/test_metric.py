import math

import pytest

from core.models.metric import MeanMetric, MetricCollection, SumMetric


class TestMeanMetric:
    def test_for_empty(self):
        metric = MeanMetric()
        assert math.isnan(metric.compute())

    def test_weighted_mean(self):
        metric = MeanMetric()
        metric.update(1.0, weight=1)
        metric.update(3.0, weight=3)
        assert metric.compute() == pytest.approx((1.0 + 9.0) / 4)

    def test_reset_returns_to_empty(self):
        metric = MeanMetric()
        metric.update(2.0)
        metric.reset()
        assert math.isnan(metric.compute())


class TestSumMetric:
    def test_empty_sum_is_zero(self):
        assert SumMetric().compute() == 0.0

    def test_weighted_sum(self):
        metric = SumMetric()
        metric.update(2.0)
        metric.update(3.0, weight=2)
        assert metric.compute() == pytest.approx(2.0 + 6.0)


class TestMetricCollection:
    def test_update_and_compute_apply_shared_weight(self):
        collection = MetricCollection({"a": MeanMetric(), "b": MeanMetric()})
        collection.update({"a": 1.0, "b": 2.0}, weight=1)
        collection.update({"a": 3.0, "b": 4.0}, weight=3)
        assert collection.compute() == {
            "a": pytest.approx((1.0 + 9.0) / 4),
            "b": pytest.approx((2.0 + 12.0) / 4),
        }

    def test_update_rejects_mismatched_keys(self):
        collection = MetricCollection({"a": MeanMetric()})
        with pytest.raises(KeyError):
            collection.update({"a": 1.0, "b": 2.0})

    def test_reset_clears_every_metric(self):
        collection = MetricCollection({"a": SumMetric()})
        collection.update({"a": 5.0})
        collection.reset()
        assert collection.compute() == {"a": 0.0}
