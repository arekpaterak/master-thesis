from __future__ import annotations

from pydantic import BaseModel, computed_field

from core.models.metric import MeanMetric, MetricCollection, SumMetric
from core.models.module import EpochAccumulator


class BatchTrainSummary(BaseModel):
    model_config = {"frozen": True}

    batch_size: int
    loss: float
    reward_best: float
    reward_mean: float
    entropy: float
    entropy_norm: float
    log_p: float
    reward_std: float
    improving_move_rate: float
    unique_actions_ratio: float
    diversity_overlap: float
    unique_rollouts: float
    embeddings_cos_sim: float
    objective_start: float
    objective_before: float
    objective_after: float
    grad_norm: float
    batch_time_sec: float
    instances_per_sec: float

    @computed_field
    @property
    def improvement(self) -> float:
        return self.objective_start - self.objective_after

    @computed_field
    @property
    def relative_improvement(self) -> float:
        if self.objective_start == 0.0:
            return 0.0
        return self.improvement / self.objective_start


class EpochTrainSummary(BaseModel):
    model_config = {"frozen": True}

    epoch: int
    loss: float
    reward_best: float
    reward_mean: float
    entropy: float
    entropy_norm: float
    log_p: float
    reward_std: float
    improving_move_rate: float
    unique_actions_ratio: float
    diversity_overlap: float
    unique_rollouts: float
    embeddings_cos_sim: float
    objective_start: float
    objective_before: float
    objective_after: float
    grad_norm: float
    batch_time_sec: float
    epoch_time_sec: float
    instances_per_sec: float

    @computed_field
    @property
    def improvement(self) -> float:
        return self.objective_start - self.objective_after

    @computed_field
    @property
    def relative_improvement(self) -> float:
        if self.objective_start == 0.0:
            return 0.0
        return self.improvement / self.objective_start


class EpochTrainAccumulator(EpochAccumulator):
    def __init__(self) -> None:
        self._means = MetricCollection(
            {
                name: MeanMetric()
                for name in (
                    "loss",
                    "reward_best",
                    "reward_mean",
                    "entropy",
                    "entropy_norm",
                    "log_p",
                    "reward_std",
                    "improving_move_rate",
                    "unique_actions_ratio",
                    "diversity_overlap",
                    "unique_rollouts",
                    "embeddings_cos_sim",
                    "objective_start",
                    "objective_before",
                    "objective_after",
                    "grad_norm",
                )
            }
        )
        self._instances = SumMetric()
        self._batches = SumMetric()
        self._batch_time = SumMetric()

    def update(self, summary: BatchTrainSummary) -> None:
        n = summary.batch_size
        self._instances.update(n)
        self._batches.update(1)
        self._batch_time.update(summary.batch_time_sec)
        self._means.update(
            {
                "loss": summary.loss,
                "reward_best": summary.reward_best,
                "reward_mean": summary.reward_mean,
                "entropy": summary.entropy,
                "entropy_norm": summary.entropy_norm,
                "log_p": summary.log_p,
                "reward_std": summary.reward_std,
                "improving_move_rate": summary.improving_move_rate,
                "unique_actions_ratio": summary.unique_actions_ratio,
                "diversity_overlap": summary.diversity_overlap,
                "unique_rollouts": summary.unique_rollouts,
                "embeddings_cos_sim": summary.embeddings_cos_sim,
                "objective_start": summary.objective_start,
                "objective_before": summary.objective_before,
                "objective_after": summary.objective_after,
                "grad_norm": summary.grad_norm,
            },
            n,
        )

    def complete(self, *, epoch: int) -> EpochTrainSummary:
        total_time = self._batch_time.compute()
        return EpochTrainSummary(
            epoch=epoch,
            **self._means.compute(),
            batch_time_sec=total_time / self._batches.compute(),
            epoch_time_sec=total_time,
            instances_per_sec=self._instances.compute() / total_time,
        )
