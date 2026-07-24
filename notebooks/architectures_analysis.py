import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from core.models.nds.config import (
        JointModelConfig,
        SplitModelConfig,
    )
    from core.models.nds.decoders.config import (
        AutoregressiveDecoderConfig,
        SimpleOnePassDecoderConfig,
        SlotDecoderConfig,
    )
    from core.models.nds.encoders.config import JointEncoderConfig, SplitEncoderConfig
    from core.models.nds.factory import build_model

    ENCODER_MODES = ("joint", "split")
    # Zero-arg factories so a variant can carry non-default config (e.g. ar-no-gru).
    DECODER_CONFIGS = {
        "ar": AutoregressiveDecoderConfig,
        "ar-no-gru": lambda: AutoregressiveDecoderConfig(use_hidden_decoding_state=False),
        "simple": SimpleOnePassDecoderConfig,
        "slot": SlotDecoderConfig,
    }
    # Encoder-layer variants compared purely by parameter count.
    # "classic" = defaults (InstanceNorm, post-norm, ReLU FF).
    # "modern" = RMSNorm + pre-norm + SwiGLU FF.
    ENCODER_SETTINGS = {
        "classic": {},
        "modern": dict(norm_type="rms", pre_norm=True, ff_activation="swiglu"),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Architectures analysis

    Trainable-parameter comparison across three axes:

    - **encoder** -- `joint` (re-encodes instance + solution each step) vs `split`
      (encodes the instance once, conditions on the solution through a `fusion` block).
    - **decoder** -- `ar`, `ar-no-gru` (autoregressive without the GRU decoding
      state), `simple`, `slot`.
    - **encoder settings** -- `classic` (InstanceNorm / post-norm / ReLU) vs `modern`
      (RMSNorm / pre-norm / SwiGLU).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)
    return


@app.cell
def _():
    problem_dd = mo.ui.dropdown(["cvrp", "tsp"], value="cvrp", label="Problem")
    hidden_dd = mo.ui.dropdown(["64", "128", "256"], value="128", label="Hidden dim")
    heads_dd = mo.ui.dropdown(["4", "8"], value="8", label="Heads")
    ff_dd = mo.ui.dropdown(["256", "512", "1024"], value="512", label="FF dim")
    pre_slider = mo.ui.slider(0, 4, value=2, label="Encoder pre-attention layers")
    post_slider = mo.ui.slider(0, 4, value=2, label="Encoder post-attention layers")

    controls = mo.vstack(
        [
            mo.hstack([problem_dd, hidden_dd, heads_dd, ff_dd], justify="start"),
            mo.hstack([pre_slider, post_slider], justify="start"),
        ]
    )
    controls
    return ff_dd, heads_dd, hidden_dd, post_slider, pre_slider, problem_dd


@app.cell
def _(ff_dd, heads_dd, hidden_dd, post_slider, pre_slider, problem_dd):
    problem = problem_dd.value
    dims = dict(
        hidden_dim=int(hidden_dd.value),
        ff_dim=int(ff_dd.value),
        num_heads=int(heads_dd.value),
        num_pre=pre_slider.value,
        num_post=post_slider.value,
    )
    return dims, problem


@app.cell
def _():
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def build_variant(encoder_mode, decoder_type, settings, problem, *, dims):
        decoder = DECODER_CONFIGS[decoder_type]()
        shared = dict(
            problem_name=problem,
            hidden_dim=dims["hidden_dim"],
            ff_dim=dims["ff_dim"],
            num_heads=dims["num_heads"],
            decoder=decoder,
        )
        if encoder_mode == "joint":
            encoder = JointEncoderConfig(
                num_pre_attention_layers=dims["num_pre"],
                num_post_attention_layers=dims["num_post"],
                **ENCODER_SETTINGS[settings],
            )
            model_params = JointModelConfig(encoder=encoder, **shared)
        else:
            # Route encoding is a CVRP-only signal; the split encoder rejects it for TSP.
            # The instance encoder has no pre/post split (structural layers are always
            # off there), so the two slider values collapse into one attention block.
            route_extra = {} if problem == "cvrp" else dict(with_route_encoding=False)
            encoder = SplitEncoderConfig(
                num_instance_attention_layers=dims["num_pre"] + dims["num_post"],
                **ENCODER_SETTINGS[settings],
                **route_extra,
            )
            model_params = SplitModelConfig(encoder=encoder, **shared)
        return build_model(model_params)

    def variant_stats(encoder_mode, decoder_type, settings, problem, *, dims):
        model = build_variant(encoder_mode, decoder_type, settings, problem, dims=dims)
        children = dict(model.named_children())
        encoder = count_params(children["encoder"])
        fusion = count_params(children["fusion"]) if "fusion" in children else 0
        decoder = count_params(children["decoder"])
        return {
            "encoder_mode": encoder_mode,
            "decoder": decoder_type,
            "settings": settings,
            "encoder": encoder,
            "fusion": fusion,
            "decoder_params": decoder,
            "total": encoder + fusion + decoder,
        }

    return (variant_stats,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Parameter count by variant
    """)
    return


@app.cell
def _(dims, problem, variant_stats):
    params_df = pl.DataFrame(
        [
            variant_stats(encoder_mode, decoder_type, settings, problem, dims=dims)
            for encoder_mode in ENCODER_MODES
            for decoder_type in DECODER_CONFIGS
            for settings in ENCODER_SETTINGS
        ]
    ).sort("encoder_mode", "decoder", "settings")
    params_df
    return (params_df,)


@app.cell
def _(params_df):
    totals = {(r["encoder_mode"], r["decoder"], r["settings"]): r for r in params_df.to_dicts()}
    return (totals,)


@app.cell
def _(problem, totals):
    _decoders = list(DECODER_CONFIGS)
    _combos = [(m, s) for m in ENCODER_MODES for s in ENCODER_SETTINGS]
    _x = np.arange(len(_decoders))
    _width = 0.2

    fig_total, ax_total = plt.subplots(figsize=(9, 4.5))
    for _i, (_mode, _settings) in enumerate(_combos):
        _vals = [totals[(_mode, d, _settings)]["total"] for d in _decoders]
        _offset = (_i - (len(_combos) - 1) / 2) * _width
        _bars = ax_total.bar(_x + _offset, _vals, _width, label=f"{_mode} / {_settings}")
        ax_total.bar_label(_bars, fmt="{:,.0f}", fontsize=7, rotation=90, padding=2)

    ax_total.set_xticks(_x, _decoders)
    ax_total.set_xlabel("decoder")
    ax_total.set_ylabel("trainable parameters")
    ax_total.set_title(f"Total parameters ({problem})")
    ax_total.legend(title="encoder / settings", fontsize=8)
    ax_total.margins(y=0.2)
    fig_total
    return


@app.cell
def _(params_df, problem):
    _components = ["encoder", "fusion", "decoder_params"]
    _settings = list(ENCODER_SETTINGS)

    fig_break, axes_break = plt.subplots(1, len(_settings), figsize=(11, 4.5), sharey=True)
    for _ax, _setting in zip(axes_break, _settings):
        _sub = params_df.filter(pl.col("settings") == _setting).sort("encoder_mode", "decoder")
        _labels = [f"{r['encoder_mode']}/{r['decoder']}" for r in _sub.to_dicts()]
        _pos = np.arange(len(_labels))
        _bottom = np.zeros(len(_labels))
        for _component in _components:
            _vals = np.array(_sub[_component].to_list())
            _ax.bar(_pos, _vals, bottom=_bottom, label=_component)
            _bottom = _bottom + _vals
        _ax.set_xticks(_pos, _labels, rotation=40, ha="right")
        _ax.set_title(_setting)

    axes_break[0].set_ylabel("trainable parameters")
    axes_break[0].legend(fontsize=8)
    fig_break.suptitle(f"Parameter breakdown by component ({problem})")
    fig_break.tight_layout()
    fig_break
    return


if __name__ == "__main__":
    app.run()
