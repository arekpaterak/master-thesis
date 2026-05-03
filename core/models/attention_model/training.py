import time
from collections import OrderedDict
from collections.abc import Callable

import torch
from pydantic import BaseModel
from tqdm.auto import tqdm

from core.models.attention_model.model import AttentionModel, Rollout
from core.models.attention_model.training_helpers import one_sided_paired_ttest, sync_baseline_
from core.problems.instance import BaseProblemInstance


class BatchSummary(BaseModel):
    model_config = {"frozen": True}

    loss: float
    advantage: float
    entropy: float
    entropy_norm: float
    grad_norm: float
    log_p: float


class EpochSummary(BaseModel):
    model_config = {"frozen": True}

    epoch: int
    loss: float
    advantage: float
    entropy: float
    entropy_norm: float
    grad_norm: float
    log_p: float
    obj_model: float
    obj_baseline: float
    baseline_updated: bool
    instances_per_sec: float


def _training_step(
    *,
    model: AttentionModel,
    baseline: AttentionModel,
    optimizer: torch.optim.Optimizer,
    instance: BaseProblemInstance,
) -> BatchSummary:
    optimizer.zero_grad()

    rollout: Rollout = model.rollout(instance, greedy=False)

    with torch.no_grad():
        baseline_rollout = baseline.rollout(instance, greedy=True)

    obj_model = rollout.to_solution(instance).objective()
    obj_baseline = baseline_rollout.to_solution(instance).objective()
    advantage = (obj_model - obj_baseline).detach()
    loss = (advantage * rollout.overall_log_p).mean()

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return BatchSummary(
        loss=loss.item(),
        advantage=advantage.mean().item(),
        entropy=rollout.entropy or 0.0,
        entropy_norm=rollout.entropy_norm or 0.0,
        grad_norm=grad_norm.item(),
        log_p=rollout.log_p_of_actions.mean().item(),
    )


def configure_train_epoch(
    *,
    instance_sampler: Callable[[int], BaseProblemInstance],
    batches_per_epoch: int = 2500,
    batch_size: int = 512,
    alpha: float = 0.05,
    eval_n_instances: int = 10_000,
    eval_batch_size: int = 512,
    on_step_end: Callable[[BatchSummary], None] | None = None,
) -> Callable[..., EpochSummary]:
    def train_epoch(
        *,
        epoch: int,
        model: AttentionModel,
        optimizer: torch.optim.Optimizer,
        baseline: AttentionModel,
    ) -> EpochSummary:
        device = next(model.parameters()).device
        model.train()

        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_entropy = 0.0
        total_entropy_norm = 0.0
        total_grad_norm = 0.0
        total_advantage = 0.0
        total_log_p = 0.0

        step_bar = tqdm(
            range(1, batches_per_epoch + 1),
            desc=f"Epoch {epoch}",
            unit="batch",
            leave=False,
        )
        for _ in step_bar:
            instance = instance_sampler(batch_size).to(device)
            batch_summary = _training_step(
                model=model,
                baseline=baseline,
                optimizer=optimizer,
                instance=instance,
            )

            total_loss += batch_summary.loss
            total_entropy += batch_summary.entropy
            total_entropy_norm += batch_summary.entropy_norm
            total_grad_norm += batch_summary.grad_norm
            total_advantage += batch_summary.advantage
            total_log_p += batch_summary.log_p

            step_bar.set_postfix(
                OrderedDict(
                    loss=f"{batch_summary.loss:.4f}",
                    entropy=f"{batch_summary.entropy:.3f}",
                    entropy_norm=f"{batch_summary.entropy_norm:.3f}",
                )
            )

            if on_step_end is not None:
                on_step_end(batch_summary)

        elapsed_time = time.perf_counter() - epoch_start

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

        return EpochSummary(
            epoch=epoch,
            loss=total_loss / batches_per_epoch,
            advantage=total_advantage / batches_per_epoch,
            entropy=total_entropy / batches_per_epoch,
            entropy_norm=total_entropy_norm / batches_per_epoch,
            grad_norm=total_grad_norm / batches_per_epoch,
            log_p=total_log_p / batches_per_epoch,
            obj_model=test.obj_model_mean,
            obj_baseline=test.obj_baseline_mean,
            baseline_updated=test.improved,
            instances_per_sec=(batches_per_epoch * batch_size) / elapsed_time,
        )

    return train_epoch
