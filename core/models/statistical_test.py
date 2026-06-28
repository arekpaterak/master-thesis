from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from scipy import stats


class OneSidedPairedTTestResult(NamedTuple):
    improved: bool
    model_mean: float
    baseline_mean: float
    p_value: float


def one_sided_paired_ttest[M](
    *,
    model: M,
    baseline: M,
    metric: Callable[[M], torch.Tensor],
    alpha: float = 0.05,
) -> OneSidedPairedTTestResult:
    """One-sided paired t-test: is ``model`` significantly better than ``baseline``?

    ``metric`` maps a model to a per-instance, lower-is-better score ``(num_instances,)``.
    It is applied to both models, which must therefore be scored on the same instances so
    the samples are paired. ``improved`` is True when the model's score is significantly
    lower than the baseline's at the given ``alpha``.
    """
    values_model = metric(model).cpu().numpy()
    values_baseline = metric(baseline).cpu().numpy()
    t, p = stats.ttest_rel(values_model, values_baseline)
    p_one_sided = p / 2 if t < 0 else 1.0
    return OneSidedPairedTTestResult(
        improved=bool(p_one_sided < alpha),
        model_mean=float(values_model.mean()),
        baseline_mean=float(values_baseline.mean()),
        p_value=float(p_one_sided),
    )
