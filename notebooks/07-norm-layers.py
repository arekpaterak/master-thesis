# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.0.0",
#     "numpy",
#     "matplotlib",
#     "pytest",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Normalization layers: four axis choices and one extra formula variant
    """)
    return


@app.cell(hide_code=True)
def _():
    from enum import Enum

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from matplotlib.colors import ListedColormap

    class Framing(Enum):
        VISION = "vision"
        TRANSFORMER = "transformer"

    _ = torch.manual_seed(0)
    return Framing, ListedColormap, mo, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Batch**, **Layer**, **Instance** and **Group** normalizations all compute the same thing:

    $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}\,\gamma + \beta,$$

    where
    - $x$ is the input activation;
    - $\mu$ and $\sigma^2$ are the mean and variance
    reduced over the chosen axes;
    - $\varepsilon$ is a small constant for numerical
    stability;
    - $\gamma$ and $\beta$ are **learnable** scale and shift;
    - $\hat{x}$ is the normalised output.

    They differ **only** in which axes the mean $\mu$ and variance $\sigma^2$
    are reduced over.

    ---

    **RMSNorm** (Root Mean Square Normalization) uses the same axes as LayerNorm
    but drops the centering step, replacing $\mu$ and $\sigma^2$ with just the
    root mean square:

    $$\hat{x} = \frac{x}{\sqrt{\frac{1}{n}\sum_i x_i^2 + \varepsilon}}\,\gamma$$

    No mean subtraction, **no bias** $\beta$. The hypothesis is that
    re-centering adds little and removing it both simplifies and speeds up
    the layer.

    ---

    Pick a *framing* and a *norm* below; everything in the
    notebook is derived from these choices.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task-dependant framing

    A gotcha this notebook makes explicit: the name *LayerNorm* denotes
    **different** reductions by convention.
    - In vision, it pools over
    $(C, H, W)$.
    - In transformers, it pools over the feature dim $D$ **per
    token** and never over the $N$ tokens. The token axis is generic:
    tokenised words in NLP, patches in ViT, nodes in graph attention (e.g. for
    combinatorial optimisation), or any set of elements processed
    jointly by attention.

    The framing switch therefore changes which
    axes each name maps to.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why each norm is named the way it is

    - **Batch**Norm — pools across the *batch*: the statistic is estimated
      from all instances seen together in one training step. The name is
      the axis it reads.
    - **Layer**Norm — normalizes everything a *layer* produces for one
      input. Coined for RNNs: the full hidden-state vector at one timestep
      is "what the layer produced", so normalize all of it. In vision
      that is the full $C \times H \times W$ feature map for one image;
      in a transformer it is the $D$-dimensional embedding of one token
      (each token is its own independent input to the layer). The
      per-token behavior is a consequence, not the definition.
    - **Instance**Norm — the name emphasizes *independence from the batch*:
      statistics are computed within each instance without looking at any
      other instance. Originally introduced for style transfer. Note the
      tension with our use of "instance" for batch elements: InstanceNorm
      does not pool the whole instance — it keeps channels separate and
      only averages over spatial positions $(H, W)$ within each
      (instance, channel) pair.
    - **Group**Norm — divides channels into *groups* and normalizes within
      each group. $G=1$ recovers LayerNorm; $G=C$ recovers InstanceNorm.
    - **RMS**Norm — named after the *formula*, not the axes: replaces the
      standard-deviation denominator with the root mean square, dropping
      the centering step entirely.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore normalization variants
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    framing = mo.ui.dropdown(
        options=["vision (B, C, H, W)", "transformer (B, N, D)"],
        value="vision (B, C, H, W)",
        label="Framing",
    )
    framing
    return (framing,)


@app.cell
def _(Framing, framing, torch):
    framing_kind = Framing.VISION if framing.value.startswith("vision") else Framing.TRANSFORMER

    if framing_kind == Framing.VISION:
        _B, _C, _H, _W = 4, 6, 4, 4
        x = torch.randn(_B, _C, _H, _W)
        feature_size = _C
        shape_label = f"(B={_B}, C={_C}, H={_H}, W={_W})"
    else:
        _B, _N, _D = 4, 8, 6
        x = torch.randn(_B, _N, _D)
        feature_size = _D
        shape_label = f"(B={_B}, N={_N}, D={_D})"

    divisors = [g for g in range(1, feature_size + 1) if feature_size % g == 0]
    return divisors, framing_kind, shape_label, x


@app.cell(hide_code=True)
def _(divisors, mo):
    norm_kind = mo.ui.dropdown(
        options=["batch", "instance", "group", "layer", "rms"],
        value="layer",
        label="Normalization",
    )
    num_groups = mo.ui.dropdown(
        options=[str(g) for g in divisors],
        value=str(divisors[1] if len(divisors) > 1 else divisors[0]),
        label="num_groups (GroupNorm only)",
    )
    return norm_kind, num_groups


@app.cell(hide_code=True)
def _(mo, norm_kind, num_groups):
    _widgets = [norm_kind, num_groups] if norm_kind.value == "group" else [norm_kind]
    mo.hstack(_widgets)
    return


@app.cell(hide_code=True)
def _(Framing, framing_kind, mo, norm_kind, num_groups, shape_label):
    _g = int(num_groups.value)
    if framing_kind == Framing.VISION:
        _desc = {
            "batch": "reduce over (B, H, W) → one statistic per channel",
            "layer": "reduce over (C, H, W) → one statistic per instance",
            "instance": "reduce over (H, W) → one statistic per (instance, channel)",
            "group": f"split C into {_g} groups, reduce over (C/{_g}, H, W) → "
            f"one statistic per (instance, group)",
            "rms": "Root Mean Square Normalization — same axes as LayerNorm: reduce over "
            "(C, H, W), but replaces mean+variance with RMS (no centering, no β)",
        }[norm_kind.value]
    else:
        _desc = {
            "batch": "reduce over (B, N) → one statistic per feature",
            "layer": "reduce over (D,) per token → one statistic per (instance, token)",
            "instance": "reduce over (N,) → one statistic per (instance, feature)",
            "group": f"split D into {_g} groups, reduce over (D/{_g},) per token → "
            f"one statistic per (instance, position, group)",
            "rms": "Root Mean Square Normalization — same axes as LayerNorm: reduce over "
            "(D,) per token, but replaces mean+variance with RMS (no centering, no β)",
        }[norm_kind.value]
    mo.md(f"**Input** `{shape_label}`\n\n**Selected norm**: {norm_kind.value}\n\n{_desc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The primitive below implements only the normalization step — no learnable
    $\gamma$ or $\beta$. This notebook focuses on **which axes** each norm
    reduces over; the affine step gives no additional insight. To compare the primitive against `nn.*` modules, we
    disable their affine parameters (`affine=False`).
    """)
    return


@app.cell
def _(Framing, torch):
    # The single primitive. Every named norm is this function with different
    # `axes` (GroupNorm first reshapes to expose a group axis, then calls it).
    def normalize(x: torch.Tensor, axes: tuple[int, ...], eps: float = 1e-5) -> torch.Tensor:
        mean = x.mean(dim=axes, keepdim=True)
        var = x.var(dim=axes, unbiased=False, keepdim=True)  # biased, like nn.*
        return (x - mean) / torch.sqrt(var + eps)

    def normalize_rms(x: torch.Tensor, axes: tuple[int, ...], eps: float = 1e-5) -> torch.Tensor:
        rms = torch.sqrt((x**2).mean(dim=axes, keepdim=True) + eps)
        return x / rms

    def normalize_named(
        x: torch.Tensor, kind: str, framing: Framing, G: int, eps: float = 1e-5
    ) -> torch.Tensor:
        """Apply a named norm by choosing axes (and reshaping for group)."""
        if kind == "rms":
            axes = (1, 2, 3) if framing == Framing.VISION else (2,)
            return normalize_rms(x, axes=axes, eps=eps)
        if kind == "group":
            if framing == Framing.VISION:
                B, C, H, W = x.shape
                xg = x.reshape(B, G, C // G, H, W)
                out = normalize(xg, axes=(2, 3, 4), eps=eps)
                return out.reshape(B, C, H, W)
            else:
                B, N, Dc = x.shape
                xg = x.reshape(B, N, G, Dc // G)
                out = normalize(xg, axes=(3,), eps=eps)  # over feature subgroup, per token
                return out.reshape(B, N, Dc)
        if framing == Framing.VISION:
            axes = {"batch": (0, 2, 3), "layer": (1, 2, 3), "instance": (2, 3)}[kind]
        else:
            axes = {"batch": (0, 1), "layer": (2,), "instance": (1,)}[kind]
        return normalize(x, axes=axes, eps=eps)

    return normalize, normalize_named


@app.cell(hide_code=True)
def _(
    Framing,
    ListedColormap,
    framing_kind,
    norm_kind,
    np,
    num_groups,
    plt,
    x,
):
    # Visual: which cells share a statistic? Two cells are in the same group iff
    # they agree on every *kept* (non-reduced) coordinate. We render a few
    # instances as (feature x spatial) grids and colour by group id.
    _G = int(num_groups.value)
    kind = norm_kind.value
    n_show = min(3, x.shape[0])

    if framing_kind == Framing.VISION:
        feat = x.shape[1]
        spat = int(np.prod(x.shape[2:]))
    else:
        feat = x.shape[2]
        spat = x.shape[1]  # N tokens sit on the "spatial" axis

    def group_key(n: int, f: int, s: int) -> tuple[int, ...]:
        # returns a hashable key; cells sharing a key share a statistic
        fg = f // (feat // _G)
        if framing_kind == Framing.VISION:
            return {
                "batch": (f,),  # pool over n, s
                "layer": (n,),  # pool over f, s
                "rms": (n,),  # same axes as layer
                "instance": (n, f),  # pool over s
                "group": (n, fg),  # pool over feature-subgroup, s
            }[kind]
        else:
            return {
                "batch": (f,),  # pool over n, s(=N)
                "layer": (n, s),  # pool over f(=D), per token
                "rms": (n, s),  # same axes as layer
                "instance": (n, f),  # pool over s(=N)
                "group": (n, s, fg),  # pool over feature-subgroup, per token
            }[kind]

    keys = {}
    grids = np.zeros((n_show, feat, spat), dtype=int)
    for n in range(n_show):
        for f in range(feat):
            for s in range(spat):
                _k = group_key(n, f, s)
                if _k not in keys:
                    keys[_k] = len(keys)
                grids[n, f, s] = keys[_k]

    n_colors = len(keys)
    _palette = [
        *plt.get_cmap("tab20").colors,
        *plt.get_cmap("tab20b").colors,
        *plt.get_cmap("tab20c").colors,
    ]
    cmap = ListedColormap(_palette[: max(n_colors, 1)])

    fig_groups, axs = plt.subplots(1, n_show, figsize=(2.4 * n_show, 3.2), squeeze=False)
    feat_name = "channel C" if framing_kind == Framing.VISION else "feature D"
    spat_name = "spatial H·W" if framing_kind == Framing.VISION else "tokens N"
    for n in range(n_show):
        ax = axs[0, n]
        ax.imshow(grids[n], cmap=cmap, vmin=0, vmax=max(n_colors - 1, 1), aspect="auto")
        ax.set_title(f"instance {n}", fontsize=10)
        ax.set_xlabel(spat_name, fontsize=8)
        if n == 0:
            ax.set_ylabel(feat_name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for boundary in np.arange(0.5, feat - 0.5, 1.0):
            ax.axhline(boundary, color="white", linestyle="--", linewidth=0.5, alpha=0.6)
    fig_groups.suptitle(
        f"{kind} norm — cells sharing a colour are normalized together "
        f"({n_colors} statistics over {n_show} shown instances)",
        fontsize=10,
    )
    fig_groups.tight_layout()
    fig_groups
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reading the figure: **BatchNorm** colours are shared *across* instance
    panels (it pools the batch axis); every other norm keeps statistics
    *within* an instance. Slide `num_groups` from `1` to the feature size and
    GroupNorm sweeps continuously from LayerNorm-like (one group) to
    InstanceNorm-like (one group per feature) — that interpolation is the
    point of the next section.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Equivalences
    """)
    return


@app.cell
def _(Framing, framing_kind, nn, normalize_named, num_groups, x):
    # PyTorch module zoo: build the canonical nn.* module for each norm under
    # the current framing, with affine disabled so outputs match the primitive.
    _G = int(num_groups.value)
    eps = 1e-5
    zoo = {}

    if framing_kind == Framing.VISION:
        _C = x.shape[1]
        _H, _W = x.shape[2], x.shape[3]
        zoo["batch"] = nn.BatchNorm2d(_C, eps=eps, affine=False)(x)
        zoo["layer"] = nn.LayerNorm([_C, _H, _W], eps=eps, elementwise_affine=False)(x)
        zoo["instance"] = nn.InstanceNorm2d(_C, eps=eps, affine=False)(x)
        zoo["group"] = nn.GroupNorm(_G, _C, eps=eps, affine=False)(x)
        zoo["rms"] = nn.RMSNorm([_C, _H, _W], eps=eps, elementwise_affine=False)(x)
    else:
        _B, _N, _D = x.shape
        xt = x.permute(0, 2, 1)  # (B, D, N) for the 1d batch/instance modules
        zoo["batch"] = nn.BatchNorm1d(_D, eps=eps, affine=False)(xt).permute(0, 2, 1)
        zoo["layer"] = nn.LayerNorm(_D, eps=eps, elementwise_affine=False)(x)
        zoo["instance"] = nn.InstanceNorm1d(_D, eps=eps, affine=False)(xt).permute(0, 2, 1)
        # transformer GroupNorm = per-token: fold N into batch so N is NOT pooled
        xg = x.reshape(_B * _N, _D, 1)
        zoo["group"] = nn.GroupNorm(_G, _D, eps=eps, affine=False)(xg).reshape(_B, _N, _D)
        zoo["rms"] = nn.RMSNorm(_D, eps=eps, elementwise_affine=False)(x)

    # primitive outputs for the same norms
    prim = {_k: normalize_named(x, _k, framing_kind, _G, eps) for _k in zoo}
    return eps, prim, zoo


@app.cell
def _(mo, prim, torch, zoo):
    # Universal check: every nn.* module equals the hand-rolled primitive.
    rows = []
    for _k in zoo:
        match = torch.allclose(zoo[_k], prim[_k], atol=1e-5)
        max_abs = (zoo[_k] - prim[_k]).abs().max().item()
        rows.append(f"| `{_k}` | {'✅' if match else '❌'} | {max_abs:.2e} |")
    table = "| norm | nn.* == primitive | max abs diff |\n|---|---|---|\n" + "\n".join(rows)
    _note = "> `affine=False` on all modules — matching the primitive which has no $\\gamma$ or $\\beta$."
    mo.md(
        "### Every PyTorch module is the primitive with different axes\n\n" + table + "\n\n" + _note
    )
    return


@app.cell
def _(Framing, eps, framing_kind, mo, nn, torch, x):
    # Framing-specific equivalences between *different* named modules.
    lines = []
    if framing_kind == Framing.VISION:
        _C, _H, _W = x.shape[1], x.shape[2], x.shape[3]
        gn1 = nn.GroupNorm(1, _C, eps=eps, affine=False)(x)
        ln = nn.LayerNorm([_C, _H, _W], eps=eps, elementwise_affine=False)(x)
        lines.append(("GroupNorm(G=1) == LayerNorm(C,H,W)", torch.allclose(gn1, ln, atol=1e-5)))

        gnC = nn.GroupNorm(_C, _C, eps=eps, affine=False)(x)
        inrm = nn.InstanceNorm2d(_C, eps=eps, affine=False)(x)
        lines.append(("GroupNorm(G=C) == InstanceNorm2d", torch.allclose(gnC, inrm, atol=1e-5)))
    else:
        _B, _N, _D = x.shape
        ln = nn.LayerNorm(_D, eps=eps, elementwise_affine=False)(x)
        gn1 = nn.GroupNorm(1, _D, eps=eps, affine=False)(x.reshape(_B * _N, _D, 1)).reshape(
            _B, _N, _D
        )
        lines.append(("LayerNorm(D) == GroupNorm(1) per token", torch.allclose(ln, gn1, atol=1e-5)))

        # transpose duality on a plain (B, D) matrix
        y = torch.randn(_B, _D)
        _bn = nn.BatchNorm1d(_D, eps=eps, affine=False)(y)
        ln2 = nn.LayerNorm(_B, eps=eps, elementwise_affine=False)(y.T).T
        lines.append(("BatchNorm(y) == LayerNorm(yᵀ)ᵀ", torch.allclose(_bn, ln2, atol=1e-5)))

    body = "\n".join(f"- {'✅' if ok else '❌'} `{name}`" for name, ok in lines)
    note = (
        "Equivalence holds on the *normalized output*. The affine parameters "
        "differ in shape (GroupNorm/Instance are per-channel `(C,)`, while "
        "LayerNorm over `(C,H,W)` is elementwise `(C,H,W)`), so the asserts "
        "disable affine on both sides."
    )
    mo.md("### Cross-module equivalences (current framing)\n\n" + body + "\n\n> " + note)
    return


@app.cell
def _(eps, nn, normalize, torch):
    def test_groupnorm_endpoints_vision():
        C, H, W = 6, 4, 4
        xv = torch.randn(2, C, H, W)
        gn1 = nn.GroupNorm(1, C, eps=eps, affine=False)(xv)
        ln = nn.LayerNorm([C, H, W], eps=eps, elementwise_affine=False)(xv)
        assert torch.allclose(gn1, ln, atol=1e-5)
        gnC = nn.GroupNorm(C, C, eps=eps, affine=False)(xv)
        inrm = nn.InstanceNorm2d(C, eps=eps, affine=False)(xv)
        assert torch.allclose(gnC, inrm, atol=1e-5)

    def test_transformer_layernorm_is_groupnorm1():
        B, N, D = 3, 5, 8
        xt = torch.randn(B, N, D)
        ln = nn.LayerNorm(D, eps=eps, elementwise_affine=False)(xt)
        gn1 = nn.GroupNorm(1, D, eps=eps, affine=False)(xt.reshape(B * N, D, 1)).reshape(B, N, D)
        assert torch.allclose(ln, gn1, atol=1e-5)

    def test_batchnorm_layernorm_transpose_duality():
        y = torch.randn(4, 7)
        bn = nn.BatchNorm1d(7, eps=eps, affine=False)(y)
        ln = nn.LayerNorm(4, eps=eps, elementwise_affine=False)(y.T).T
        assert torch.allclose(bn, ln, atol=1e-5)

    def test_primitive_matches_layernorm():
        xv = torch.randn(3, 5, 4, 4)
        ln = nn.LayerNorm([5, 4, 4], eps=eps, elementwise_affine=False)(xv)
        prim = normalize(xv, axes=(1, 2, 3), eps=eps)
        assert torch.allclose(ln, prim, atol=1e-5)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BatchNorm is special

    ### BatchNorm depends on batch size

    Only BatchNorm reads the batch axis, so its statistics depend on batch
    size and composition. The others are computed per-instance and are
    identical no matter how the batch is formed.

    Below: resample random batches at each size and measure the variance of
    the estimated channel mean across resamples — it shrinks as the batch
    grows, while the per-instance norms have zero such variance by construction.
    """)
    return


@app.cell
def _(plt, torch):
    # Empirical batch-size sensitivity of BatchNorm's channel-mean estimate.
    torch.manual_seed(0)
    C_dem = 16
    pop = torch.randn(4096, C_dem) * 2.0 + 1.0  # population of activations
    sizes = [2, 4, 8, 16, 32, 64, 128]
    n_resample = 1000

    stds = []
    for bs in sizes:
        means = torch.stack(
            [pop[torch.randint(0, pop.shape[0], (bs,))].mean(0) for _ in range(n_resample)]
        )  # (n_resample, C)
        stds.append(means.std(0).mean().item())  # avg over channels of the std of the estimate

    fig_batch, ax_b = plt.subplots(figsize=(5.2, 3.4))
    ax_b.plot(sizes, stds, "o-", label="BatchNorm")
    ax_b.axhline(0.0, color="gray", ls="--", lw=1, label="Layer/Instance/GroupNorm")
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(sizes)
    ax_b.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax_b.set_xlabel("Batch size")
    ax_b.set_ylabel("std of estimated channel mean")
    ax_b.set_title("BatchNorm statistics are noisier for small batches")
    ax_b.legend(fontsize=8)
    fig_batch.tight_layout()
    fig_batch
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### BatchNorm at inference is just an affine map (and can fold away)

    In `eval()` mode BatchNorm uses fixed running statistics, so it is a pure
    affine transform $y = \gamma\,(x-\mu)/\sqrt{\sigma^2+\varepsilon} + \beta$.

    A preceding `Linear` (or `Conv`) can absorb it: the two layers collapse
    into one. This is the standard *BN folding* used for inference speedups.
    """)
    return


@app.cell
def _(mo, nn, torch):
    torch.manual_seed(0)
    in_f, out_f, B = 5, 4, 32

    lin = nn.Linear(in_f, out_f)
    _bn = nn.BatchNorm1d(out_f)
    # populate non-trivial running stats, then freeze
    _bn.train()
    for _ in range(20):
        _bn(lin(torch.randn(B, in_f)))
    _bn.eval()
    lin.eval()

    xb = torch.randn(B, in_f)
    ref = _bn(lin(xb))

    # Fold: scale = gamma / sqrt(var + eps); W' = scale * W; b' = scale*(b - mu) + beta
    scale = _bn.weight / torch.sqrt(_bn.running_var + _bn.eps)
    W_fold = lin.weight * scale.unsqueeze(1)
    b_fold = scale * (lin.bias - _bn.running_mean) + _bn.bias
    folded = xb @ W_fold.T + b_fold

    ok = torch.allclose(ref, folded, atol=1e-5)
    mo.md(
        f"`Linear → BatchNorm(eval)` folded into one `Linear`: "
        f"{'✅ match' if ok else '❌ mismatch'} "
        f"(max abs diff `{(ref - folded).abs().max().item():.2e}`)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Norm | Reduces over (vision) | Reduces over (transformer) | Batch-dependent? | Running stats? |
    |---|---|---|---|---|
    | Batch | B, H, W | B, N | yes | yes |
    | Layer | C, H, W | D (per token) | no | no |
    | Instance | H, W | N (tokens) | no | no$^\dagger$ |
    | Group | C/G, H, W | D/G (per token) | no | no |
    | RMS | C, H, W | D (per token) | no | no |

    **Running stats?** — whether exponential moving averages of mean and variance are accumulated
    during training and used as fixed statistics at inference. BatchNorm needs
    them because its batch-level estimates become unreliable at small batch sizes
    (e.g. batch size 1 at inference).

    $^\dagger$ `nn.InstanceNorm` exposes `track_running_stats` (default `False`). When enabled it
    behaves like BatchNorm in this regard.

    GroupNorm generalizes the per-instance norms: `G=1` → Layer, `G=C` →
    Instance. BatchNorm stands apart because it couples examples in a batch.
    RMSNorm shares LayerNorm's axes but drops centering.

    ### References
    - Batch — Ioffe and Szegedy (2015), *Batch Normalization*, [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
    - Layer — Ba, Kiros and Hinton (2016), *Layer Normalization*, [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
    - Instance — Ulyanov, Vedaldi and Lempitsky (2016), *Instance Normalization*, [arXiv:1607.08022](https://arxiv.org/abs/1607.08022)
    - Group — Wu and He (2018), *Group Normalization*, [arXiv:1803.08494](https://arxiv.org/abs/1803.08494) (frames Layer
      and Instance as special cases of Group; source of the canonical
      comparison figure this notebook's visual reconstructs)
    - RMS — Zhang and Sennrich (2019), *Root Mean Square Layer Normalization*, [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
    """)
    return


if __name__ == "__main__":
    app.run()
