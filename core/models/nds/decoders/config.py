from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class _DecoderConfigBase(BaseModel):
    poly_dim: int = Field(default=256, title="Poly dim", description="PolyNet hidden dimension.")
    seed_dim: int = Field(default=10, title="Seed dim", description="Seed dimension.")
    with_poly: bool = Field(
        default=True,
        title="With PolyNet",
        description=(
            "Build the PolyNet seed-conditioning head used for rollout diversity. "
            "Disable it for a single-rollout, seed-independent policy (required by the "
            "greedy rollout baseline)."
        ),
    )


class AutoregressiveDecoderConfig(_DecoderConfigBase):
    type: Literal["ar"] = "ar"
    use_hidden_decoding_state: bool = Field(
        default=True,
        title="Use hidden decoding state",
        description="Use hidden decoding state, updated with a GRU.",
    )


class SimpleOnePassDecoderConfig(_DecoderConfigBase):
    type: Literal["simple"] = "simple"
    query_strategy: Literal["mean", "max", "meanmax"] = Field(
        default="mean",
        title="Query strategy",
        description=(
            "How to pool node embeddings into the cross-attention query: 'mean' uses the "
            "mean over nodes; 'max' the per-feature max; 'meanmax' concatenates the mean "
            "and max and projects back to the hidden dimension."
        ),
    )


class SlotDecoderConfig(_DecoderConfigBase):
    type: Literal["slot"] = "slot"
    num_slots: int = Field(
        default=15,
        title="Slot queries",
        description="Number of slot queries (should match deconstruction_size).",
    )
    slot_self_attention: Literal["none", "pre", "post"] = Field(
        default="post",
        title="Slot self-attention",
        description=(
            "Where to apply self-attention among slot queries: disabled, before node "
            "cross-attention, or after node cross-attention."
        ),
    )
    node_query_strategy: Literal["none", "mean", "max", "meanmax"] = Field(
        default="none",
        title="Node query strategy",
        description=(
            "Ground the learned slot queries in the instance by adding a pooled node "
            "summary before cross-attention: 'none' keeps the pure learned queries; "
            "'mean'/'max'/'meanmax' add the corresponding node pooling (broadcast over "
            "slots)."
        ),
    )


DecoderConfig = Annotated[
    AutoregressiveDecoderConfig | SimpleOnePassDecoderConfig | SlotDecoderConfig,
    Field(discriminator="type"),
]
