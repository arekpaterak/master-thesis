from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from core.models.layers.attention_scaling import AttentionScalingConfig
from core.models.layers.cross_attention import CrossAttentionConfig
from core.models.layers.decoder_query import DecoderQueryConfig
from core.models.nds.decoders.ar import AutoregressiveNDSDecoder
from core.models.nds.decoders.base import DestroyDecoder
from core.models.nds.decoders.one_pass.simple import SimpleOnePassDecoder
from core.models.nds.decoders.one_pass.slot import SlotDecoder


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
    attention_scaling: AttentionScalingConfig | None = Field(
        default=None,
        title="Attention scaling",
        description=(
            "Size-aware scaling of the logits over the nodes: node attention and the "
            "selection scorer. None disables it."
        ),
    )

    @abstractmethod
    def build(
        self, *, hidden_dim: int, num_heads: int, num_context_nodes: int = 0
    ) -> DestroyDecoder:
        """``num_context_nodes`` is the number of leading non-selectable nodes (e.g. the depot)."""
        ...

    def _shared(self, *, hidden_dim: int, num_heads: int) -> dict[str, Any]:
        return dict(
            hidden_dim=hidden_dim,
            seed_dim=self.seed_dim,
            poly_dim=self.poly_dim,
            num_heads=num_heads,
            with_poly=self.with_poly,
            attention_scaling=self.attention_scaling,
        )

    @abstractmethod
    def build(
        self, *, hidden_dim: int, num_heads: int, num_context_nodes: int = 0
    ) -> DestroyDecoder:
        """``num_context_nodes`` is the number of leading non-selectable nodes (e.g. the depot)."""
        ...

    def _shared(self, *, hidden_dim: int, num_heads: int) -> dict[str, Any]:
        return dict(
            hidden_dim=hidden_dim,
            seed_dim=self.seed_dim,
            poly_dim=self.poly_dim,
            num_heads=num_heads,
            with_poly=self.with_poly,
        )


class AutoregressiveDecoderConfig(_DecoderConfigBase):
    type: Literal["ar"] = "ar"
    use_hidden_decoding_state: bool = Field(
        default=True,
        title="Use hidden decoding state",
        description="Use hidden decoding state, updated with a GRU.",
    )

    def build(
        self, *, hidden_dim: int, num_heads: int, num_context_nodes: int = 0
    ) -> AutoregressiveNDSDecoder:
        return AutoregressiveNDSDecoder(
            **self._shared(hidden_dim=hidden_dim, num_heads=num_heads),
            use_hidden_decoding_state=self.use_hidden_decoding_state,
        )


class SimpleOnePassDecoderConfig(_DecoderConfigBase):
    type: Literal["simple"] = "simple"
    query: DecoderQueryConfig = Field(
        default_factory=DecoderQueryConfig,
        title="Query",
        description="How the cross-attention query is built from the node embeddings.",
    )
    cross_attention: CrossAttentionConfig = Field(
        default_factory=CrossAttentionConfig,
        title="Cross-attention",
        description="How the query reads the node embeddings via cross-attention.",
    )

    @model_validator(mode="after")
    def _check_query(self) -> SimpleOnePassDecoderConfig:
        if not self.query.is_configured:
            raise ValueError(
                "The query needs a pooling, a learned base or context nodes; it cannot be empty."
            )
        return self

    def build(
        self, *, hidden_dim: int, num_heads: int, num_context_nodes: int = 0
    ) -> SimpleOnePassDecoder:
        return SimpleOnePassDecoder(
            **self._shared(hidden_dim=hidden_dim, num_heads=num_heads),
            query=self.query,
            cross_attention=self.cross_attention,
            num_context_nodes=num_context_nodes,
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
    cross_attention: CrossAttentionConfig = Field(
        default_factory=CrossAttentionConfig,
        title="Cross-attention",
        description="How the slot queries read the node embeddings via cross-attention.",
    )
    query: DecoderQueryConfig = Field(
        default_factory=lambda: DecoderQueryConfig(pooling="none"),
        title="Query",
        description=(
            "How the slot queries are grounded in the instance: the node summary is added "
            "to every learned slot query before cross-attention. With no active component "
            "the slots stay pure learned queries. The 'learned' base is not allowed here "
            "-- the slot queries already are it."
        ),
    )

    @model_validator(mode="after")
    def _check_query(self) -> SlotDecoderConfig:
        if self.query.learned:
            raise ValueError("Slot query cannot be learned; the slot queries are the learned base.")
        return self

    def build(self, *, hidden_dim: int, num_heads: int, num_context_nodes: int = 0) -> SlotDecoder:
        return SlotDecoder(
            **self._shared(hidden_dim=hidden_dim, num_heads=num_heads),
            num_slots=self.num_slots,
            slot_self_attention=self.slot_self_attention,
            cross_attention=self.cross_attention,
            query=self.query,
            num_context_nodes=num_context_nodes,
        )


DecoderConfig = Annotated[
    AutoregressiveDecoderConfig | SimpleOnePassDecoderConfig | SlotDecoderConfig,
    Field(discriminator="type"),
]
