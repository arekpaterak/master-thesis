from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import nn

Pooling = Literal["none", "mean", "max", "meanmax"]
ContextHandling = Literal["merge", "separate", "exclude"]


class DecoderQueryConfig(BaseModel):
    """Configures the query a decoder builds from the node embeddings (see DecoderQuery)."""

    pooling: Pooling = Field(
        default="mean",
        title="Query pooling",
        description=(
            "How to pool the selectable nodes into the query: 'none' adds no pooling; "
            "'mean' uses the mean over nodes; 'max' the per-feature max; 'meanmax' "
            "concatenates the mean and max and projects back to the hidden dimension."
        ),
    )
    learned: bool = Field(
        default=False,
        title="Learned query base",
        description=(
            "Add a learned query vector, independent of the node contents. On its own the "
            "query is instance-blind and instance dependence flows through the attention "
            "keys/values."
        ),
    )
    context_handling: ContextHandling = Field(
        default="merge",
        title="Context node handling",
        description=(
            "How the non-selectable context nodes (e.g. the CVRP depot) enter the query: "
            "'merge' pools them together with the selectable nodes into a single summary over all nodes; 'separate' gives them their own coordinate in the query; "
            "'exclude' drops them from the query entirely. No effect for problems without "
            "context nodes (e.g. TSP)."
        ),
    )

    @property
    def is_configured(self) -> bool:
        """Whether any component could be active, independent of the problem."""
        return self.pooling != "none" or self.learned or self.context_handling == "separate"

    def is_active_for(self, *, num_context_nodes: int) -> bool:
        """Whether any component is active given how many context nodes the problem has."""
        return (
            self.pooling != "none"
            or self.learned
            or (self.context_handling == "separate" and num_context_nodes > 0)
        )


class DecoderQuery(nn.Module):
    """Builds a single query vector from up to three orthogonal components.

    The query is a learned base plus a projection of the active node summaries, so it
    always has shape (B, 1, D). The summaries come from the node embeddings via:
    - ``pooling`` over the nodes: 'none' contributes no summary; 'mean' the mean; 'max' the
      per-feature max; 'meanmax' both.
    - ``context``: how the non-selectable context nodes (e.g. the CVRP depot) enter the
      query. 'merge' pools them together with the selectable nodes, so a single summary
      spans all nodes; 'separate' gives them their own coordinate instead of diluting them
      into the symmetric pool; 'exclude' leaves them out of the query.
    - ``learned``: a learned base vector added to the result. On its own the query is
      instance-blind and instance dependence flows through the attention keys/values.

    The active summaries are concatenated and projected back to ``hidden_dim``; a single
    summary passes through unprojected. At least one component must be active.

    Args:
        hidden_dim: Node embedding dimension (D). The query is always (B, 1, D).
        pooling: Pooling over the nodes.
        context: How the context nodes enter the query. 'merge' and 'separate' both require
            the context nodes to be passed at call time.
        learned: Whether to add a learned base vector.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        pooling: Pooling = "mean",
        context: ContextHandling = "merge",
        learned: bool = False,
    ):
        super().__init__()
        num_summaries = self._num_pooling_summaries(pooling) + int(context == "separate")
        if num_summaries == 0 and not learned:
            raise ValueError("DecoderQuery needs a pooling, a context or a learned base.")
        self.pooling = pooling
        self.context = context
        self.proj = (
            nn.Linear(num_summaries * hidden_dim, hidden_dim) if num_summaries >= 2 else None
        )
        if learned:
            self.learned = nn.Parameter(torch.empty(1, hidden_dim))
            nn.init.xavier_uniform_(self.learned.unsqueeze(0))
        else:
            self.learned = None

    @classmethod
    def from_config(
        cls, config: DecoderQueryConfig, *, hidden_dim: int, num_context_nodes: int
    ) -> DecoderQuery:
        return cls(
            hidden_dim=hidden_dim,
            pooling=config.pooling,
            context=config.context_handling if num_context_nodes > 0 else "exclude",
            learned=config.learned,
        )

    def forward(
        self,
        selectable_embedded_nodes: torch.Tensor,
        *,
        context_embedded_nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the query as a projection of the node summaries plus the learned base.

        Args:
            selectable_embedded_nodes: The selectable node embeddings, shape (B, Ns, D).
            context_embedded_nodes: The context node embeddings (e.g. the CVRP depot),
                shape (B, Nc, D). Required when ``context`` is 'merge' or 'separate';
                ignored otherwise.

        Returns:
            The query, shape (B, 1, D).
        """
        query = self._project(self._summaries(selectable_embedded_nodes, context_embedded_nodes))
        if self.learned is not None:
            B, _, D = selectable_embedded_nodes.shape
            base = self.learned.unsqueeze(0).expand(B, 1, D)  # (B, 1, D)
            query = base if query is None else query + base
        assert query is not None
        return query

    def _summaries(
        self, selectable_embedded_nodes: torch.Tensor, context_embedded_nodes: torch.Tensor | None
    ) -> list[torch.Tensor]:
        if self.context == "merge":
            assert context_embedded_nodes is not None
            return self._pool(
                self._merge_for_pooling(selectable_embedded_nodes, context_embedded_nodes)
            )
        summaries = self._pool(selectable_embedded_nodes)
        if self.context == "separate":
            assert context_embedded_nodes is not None
            summaries.append(context_embedded_nodes.mean(dim=-2, keepdim=True))  # (B, 1, D)
        return summaries

    def _project(self, summaries: list[torch.Tensor]) -> torch.Tensor | None:
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]  # (B, 1, D)
        assert self.proj is not None
        return self.proj(torch.cat(summaries, dim=-1))  # (B, 1, D)

    @staticmethod
    def _merge_for_pooling(selectable: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.cat([selectable, context], dim=-2)

    def _pool(self, h_nodes: torch.Tensor) -> list[torch.Tensor]:
        match self.pooling:
            case "mean":
                return [h_nodes.mean(dim=-2, keepdim=True)]  # (B, 1, D)
            case "max":
                return [h_nodes.max(dim=-2, keepdim=True).values]  # (B, 1, D)
            case "meanmax":
                return [
                    h_nodes.mean(dim=-2, keepdim=True),
                    h_nodes.max(dim=-2, keepdim=True).values,
                ]  # 2 x (B, 1, D)
            case "none":
                return []

    @staticmethod
    def _num_pooling_summaries(pooling: Pooling) -> int:
        return {"none": 0, "mean": 1, "max": 1, "meanmax": 2}[pooling]
