from __future__ import annotations

import time
from collections.abc import Callable

import torch
from pydantic import BaseModel

from core.models.attention_model.model import AttentionModel, rollout_to_solution
from core.models.attention_model.training_helpers import (
    PairedTTestResult,
    one_sided_paired_ttest,
    sync_baseline_,
)
from core.problems.instance import BaseProblemInstance


class BatchSummary(BaseModel):
    model_config = {"frozen": True}

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

    @classmethod
    def from_step_summaries(
        cls,
        *,
        epoch: int,
        batch_summaries: list[BatchSummary],
        **_,
    ) -> EpochSummary:
        n = len(batch_summaries)
        return cls(
            epoch=epoch,
            loss=sum(s.loss for s in batch_summaries) / n,
            advantage=sum(s.advantage for s in batch_summaries) / n,
            entropy=sum(s.entropy for s in batch_summaries) / n,
            entropy_norm=sum(s.entropy_norm for s in batch_summaries) / n,
            grad_norm=sum(s.grad_norm for s in batch_summaries) / n,
            log_p=sum(s.log_p for s in batch_summaries) / n,
            instances_per_sec=sum(s.instances_per_sec for s in batch_summaries) / n,
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

    obj_model = rollout_to_solution(rollout, instance).objective()
    obj_baseline = rollout_to_solution(baseline_rollout, instance).objective()
    advantage = (obj_model - obj_baseline).detach()
    loss = (advantage * rollout.overall_log_p).mean()

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    elapsed = time.perf_counter() - start
    batch_size = instance.batch_size[0]

    return BatchSummary(
        loss=loss.item(),
        advantage=advantage.mean().item(),
        entropy=rollout.entropy or 0.0,
        entropy_norm=rollout.entropy_norm or 0.0,
        grad_norm=grad_norm.item(),
        log_p=rollout.log_p_of_actions.mean().item(),
        instances_per_sec=batch_size / elapsed,
    )


def configure_baseline_sync_callback(
    *,
    model: AttentionModel,
    baseline: AttentionModel,
    instance_sampler: Callable[[int], BaseProblemInstance],
    eval_n_instances: int = 10_000,
    eval_batch_size: int = 512,
    alpha: float = 0.05,
    on_test_end: Callable[[PairedTTestResult], None] | None = None,
) -> Callable[[EpochSummary], None]:
    def on_epoch_end(_: EpochSummary) -> None:
        device = next(model.parameters()).device
        test = one_sided_paired_ttest(
            model,
            baseline,
            instance_sampler,
            n_instances=eval_n_instances,
            batch_size=eval_batch_size,
            alpha=alpha,
            device=device,
        )
        if test.improved:
            sync_baseline_(baseline, model)
        if on_test_end is not None:
            on_test_end(test)

    return on_epoch_end
