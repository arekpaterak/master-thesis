from __future__ import annotations

from pathlib import Path

import torch

from core.models.nds.decoders.ar import AutoregressiveNDSDecoder
from core.models.nds.encoder import NDSEncoder


def load_original_pretrained_encoder_weights(
    encoder: NDSEncoder, checkpoint_path: Path | str
) -> None:
    """Loads NDS original pretrained weights into NDSEncoder.

    The NDS checkpoint stores the full model (encoder + decoder) under "model_state_dict".
    This function extracts and remaps only the encoder weights,
    bridging the naming and layout differences between the original implementation and this codebase.
    """
    device = next(encoder.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_state_dict = checkpoint["model_state_dict"]

    out: dict[str, torch.Tensor] = {}

    # Initial embedding
    _copy(
        checkpoint_state_dict, out, "encoder.embedding_depot", "initial_embedding.depot_projection"
    )
    _copy(
        checkpoint_state_dict,
        out,
        "encoder.embedding_node",
        "initial_embedding.customer_projection",
    )

    # Pre-attention layers (encoder.layers.{i} -> pre_attention.{i})
    for i in range(2):
        _map_attention_layer(
            checkpoint_state_dict, out, f"encoder.layers.{i}", f"pre_attention.{i}"
        )

    # Neighbours encoding (encoder.mp_layers.0 -> neighbours_encoding)
    if encoder.neighbours_encoding is not None:
        src = "encoder.mp_layers.0"
        dst = "neighbours_encoding"
        _copy(
            checkpoint_state_dict, out, f"{src}.neighbour_projector", f"{dst}.neighbour_projector"
        )
        _copy(checkpoint_state_dict, out, f"{src}.neighbour_combiner", f"{dst}.neighbour_combiner")
        _copy(checkpoint_state_dict, out, f"{src}.feedforward_layer", f"{dst}.feedforward_layer")
        _copy(checkpoint_state_dict, out, f"{src}.add_and_normalize.norm", f"{dst}.norm.norm")

    # Route encoding (encoder.tour_layer -> route_encoding)
    if encoder.route_encoding is not None:
        src = "encoder.tour_layer"
        dst = "route_encoding"
        _copy(checkpoint_state_dict, out, f"{src}.tour_combiner", f"{dst}.tour_combiner")
        _copy(checkpoint_state_dict, out, f"{src}.feedforward_layer", f"{dst}.feedforward_layer")
        _copy(checkpoint_state_dict, out, f"{src}.add_and_normalize.norm", f"{dst}.norm.norm")

    # Post-attention layers (encoder.layers_2.{i} -> post_attention.{i})
    for i in range(2):
        _map_attention_layer(
            checkpoint_state_dict, out, f"encoder.layers_2.{i}", f"post_attention.{i}"
        )

    encoder.load_state_dict(out, strict=True)


def load_original_pretrained_decoder_weights(
    decoder: AutoregressiveNDSDecoder, checkpoint_path: Path | str
) -> None:
    """Loads NDS original pretrained weights into AutoregressiveNDSDecoder.

    The NDS checkpoint stores the full model (encoder + decoder) under "model_state_dict".
    This function extracts and remaps only the decoder weights.
    """
    device = next(decoder.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    nds_sd = checkpoint["model_state_dict"]

    hidden_dim = nds_sd["decoder.Wq_last.weight"].shape[0]

    out: dict[str, torch.Tensor] = {}

    # Start token
    out["start_token"] = nds_sd["start_last_node"]

    # MHA: Wq_last -> W_Q (NDS has no Q bias, so zero-fill)
    out["MHA.W_Q.weight"] = nds_sd["decoder.Wq_last.weight"]
    out["MHA.W_Q.bias"] = torch.zeros(hidden_dim, device=device)
    # Wk, Wv -> W_K, W_V (no bias on either side)
    out["MHA.W_K.weight"] = nds_sd["decoder.Wk.weight"]
    out["MHA.W_V.weight"] = nds_sd["decoder.Wv.weight"]
    # multi_head_combine -> W_O
    _copy(nds_sd, out, "decoder.multi_head_combine", "MHA.W_O")

    # GRU (key names match directly)
    if decoder.GRU is not None:
        for suffix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            out[f"GRU.{suffix}"] = nds_sd[f"decoder.GRU.{suffix}"]

    # Polynomial network: poly_layer_1/2 -> poly_net.0/2
    _copy(nds_sd, out, "decoder.poly_layer_1", "poly_net.0")
    _copy(nds_sd, out, "decoder.poly_layer_2", "poly_net.2")

    decoder.load_state_dict(out, strict=True)


def _map_attention_layer(
    nds_sd: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    nds_prefix: str,
    my_prefix: str,
) -> None:
    """Maps one NDS EncoderLayer to one AttentionLayer.

    NDS uses three separate Linear projections (Wq, Wk, Wv).
    PyTorch's MultiheadAttention stores them concatenated as in_proj_weight.
    NDS has no bias on the Q/K/V projections, so in_proj_bias is zeroed.
    """
    hidden_dim = nds_sd[f"{nds_prefix}.Wq.weight"].shape[0]

    out[f"{my_prefix}.MHA.in_proj_weight"] = torch.cat(
        [
            nds_sd[f"{nds_prefix}.Wq.weight"],
            nds_sd[f"{nds_prefix}.Wk.weight"],
            nds_sd[f"{nds_prefix}.Wv.weight"],
        ],
        dim=0,
    )
    out[f"{my_prefix}.MHA.in_proj_bias"] = torch.zeros(
        3 * hidden_dim, device=nds_sd[f"{nds_prefix}.Wq.weight"].device
    )
    _copy(nds_sd, out, f"{nds_prefix}.multi_head_combine", f"{my_prefix}.MHA.out_proj")
    _copy(nds_sd, out, f"{nds_prefix}.add_n_normalization_1.norm", f"{my_prefix}.norm1.norm")
    _copy(nds_sd, out, f"{nds_prefix}.feed_forward.W1", f"{my_prefix}.FF.0")
    _copy(nds_sd, out, f"{nds_prefix}.feed_forward.W2", f"{my_prefix}.FF.2")
    _copy(nds_sd, out, f"{nds_prefix}.add_n_normalization_2.norm", f"{my_prefix}.norm2.norm")


def _copy(
    src: dict[str, torch.Tensor],
    dst: dict[str, torch.Tensor],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    """Copies all weight/bias tensors from src_prefix to dst_prefix."""
    for suffix in ("weight", "bias"):
        src_key = f"{src_prefix}.{suffix}"
        if src_key in src:
            dst[f"{dst_prefix}.{suffix}"] = src[src_key]
