from __future__ import annotations

import time

import torch
from pydantic import BaseModel

from core.models.attention_model.model import AttentionModel, rollout_to_solution
from core.problems.instance import BaseProblemInstance


class BatchSummary(BaseModel):
    model_config = {"frozen": True}

    batch_size: int
    loss: float
    advantage: float
    entropy: float
    entropy_norm: float
    grad_norm: float
    log_p: float
    instances_per_sec: float


class EpochSummary(BaseModel):
    model_config = {"frozen": True}

    epoch: int
    loss: float
    advantage: float
    entropy: float
    entropy_norm: float
    grad_norm: float
    log_p: float
    instances_per_sec: float


class EpochTrainAccumulator:
    def __init__(self) -> None:
        self._total_instances = 0
        self._total_time = 0.0
        self._total_loss = 0.0
        self._total_advantage = 0.0
        self._total_entropy = 0.0
        self._total_entropy_norm = 0.0
        self._total_grad_norm = 0.0
        self._total_log_p = 0.0

    def update(self, summary: BatchSummary) -> None:
        n = summary.batch_size
        self._total_instances += n
        self._total_time += n / summary.instances_per_sec
        self._total_loss += summary.loss * n
        self._total_advantage += summary.advantage * n
        self._total_entropy += summary.entropy * n
        self._total_entropy_norm += summary.entropy_norm * n
        self._total_grad_norm += summary.grad_norm * n
        self._total_log_p += summary.log_p * n

    def finalize(self, *, epoch: int) -> EpochSummary:
        n = self._total_instances
        return EpochSummary(
            epoch=epoch,
            loss=self._total_loss / n,
            advantage=self._total_advantage / n,
            entropy=self._total_entropy / n,
            entropy_norm=self._total_entropy_norm / n,
            grad_norm=self._total_grad_norm / n,
            log_p=self._total_log_p / n,
            instances_per_sec=self._total_instances / self._total_time,
        )


def training_step(
    *,
    model: AttentionModel,
    baseline: AttentionModel,
    optimizer: torch.optim.Optimizer,
    instance: BaseProblemInstance,
) -> BatchSummary:
    start = time.perf_counter()
    device = next(model.parameters()).device
    instance = instance.to(device)
    model.train()

    optimizer.zero_grad()

    rollout = model.rollout(instance, greedy=False)

    with torch.no_grad():
        baseline_rollout = baseline.rollout(instance, greedy=True)

    # Minimize the cost
    obj_model = rollout_to_solution(rollout, instance).objective()
    obj_baseline = rollout_to_solution(baseline_rollout, instance).objective()
    advantage = (obj_model - obj_baseline).detach()
    # No sign flip here since we minimize the cost
    loss = (advantage * rollout.log_p).mean()

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    elapsed = time.perf_counter() - start
    batch_size = instance.batch_size[0]

    return BatchSummary(
        batch_size=batch_size,
        loss=loss.item(),
        advantage=advantage.mean().item(),
        entropy=rollout.mean_entropy,
        entropy_norm=rollout.mean_entropy_norm,
        grad_norm=grad_norm.item(),
        log_p=rollout.log_p.mean().item(),
        instances_per_sec=batch_size / elapsed,
    )
