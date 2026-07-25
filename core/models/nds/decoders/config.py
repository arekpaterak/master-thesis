from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from core.models.layers.decoder_query import Pooling


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
    query_pooling: Pooling = Field(
        default="mean",
        title="Query pooling",
        description=(
            "How to pool the nodes into the query: 'none' adds no node pooling (requires "
            "'query_learned'); 'mean' uses the mean over nodes; 'max' the per-feature max; "
            "'meanmax' concatenates the mean and max and projects back to the hidden "
            "dimension."
        ),
    )
    query_learned: bool = Field(
        default=False,
        title="Learned query base",
        description=(
            "Add a learned query vector, independent of the node contents. On its own "
            "('query_pooling' is 'none') the query is instance-blind and instance "
            "dependence flows through the attention keys/values."
        ),
    )
    query_context: bool = Field(
        default=True,
        title="Context nodes in query",
        description=(
            "Give the non-selectable context nodes (e.g. the CVRP depot) their own "
            "coordinate in the query instead of pooling them with the selectable nodes. "
            "No effect for problems without context nodes (e.g. TSP)."
        ),
    )
    skip_mha: bool = Field(
        default=False,
        title="Skip cross-attention",
        description=(
            "Skip the node cross-attention and feed the query straight to the "
            "scorer, so node scores are read directly off the (un-refined) query."
        ),
    )

    @model_validator(mode="after")
    def _check_query(self) -> SimpleOnePassDecoderConfig:
        if self.query_pooling == "none" and not self.query_learned and not self.query_context:
            raise ValueError(
                "Set query_pooling, query_learned or query_context; the query cannot be empty."
            )
        return self


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
    node_pooling: Pooling = Field(
        default="none",
        title="Node pooling",
        description=(
            "Ground the learned slot queries in the instance by adding a pooled node "
            "summary before cross-attention: 'none' keeps the pure learned queries; "
            "'mean'/'max'/'meanmax' add the corresponding node pooling (broadcast over "
            "slots)."
        ),
    )
    node_context: bool = Field(
        default=True,
        title="Context nodes in slot grounding",
        description=(
            "Ground the learned slot queries by adding the non-selectable context nodes "
            "(e.g. the CVRP depot), which the slot cross-attention and scorer otherwise "
            "never see. No effect for problems without context nodes (e.g. TSP)."
        ),
    )


DecoderConfig = Annotated[
    AutoregressiveDecoderConfig | SimpleOnePassDecoderConfig | SlotDecoderConfig,
    Field(discriminator="type"),
]
