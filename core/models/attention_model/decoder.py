import torch
import torch.nn as nn

from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.states.constructive import ConstructiveState, CVRPConstructiveState, TSPConstructiveState


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str = "tsp",
        hidden_dim: int = 128,
        num_heads: int = 8,
        C: float = 10.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.C = C

        # Learnable input placeholders for the initial decoding step
        self.v_l = nn.Parameter(torch.zeros(hidden_dim))
        self.v_f = nn.Parameter(torch.zeros(hidden_dim))

        # MODIFICATION: Contrary to the original article we project upfront, not inside Q projection in MHA
        # TODO: Extract problem specific code
        match problem_name:
            case "tsp":
                self.context_projection = nn.Linear(
                    3 * self.hidden_dim, self.hidden_dim, bias=False
                )
            case "cvrp":
                self.context_projection = nn.Linear(
                    2 * self.hidden_dim + 1, self.hidden_dim, bias=False
                )
            case _:
                raise TypeError(f"Unsupported problem type: {type(problem_name)}")

        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=False,
        )
        self.final_scorer = CompatibilityScorer(hidden_dim=hidden_dim, C=C)

    def forward(
        self,
        *,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        state: ConstructiveState,
    ) -> torch.Tensor:
        batch_size, n, _ = node_embeddings.shape

        # TODO: Extract problem specific code
        if isinstance(state, TSPConstructiveState):
            # TSP context uses first and last selected nodes.
            if state.is_initial:
                h_last = self.v_l.expand(batch_size, -1)
                h_first = self.v_f.expand(batch_size, -1)
            else:
                hidden_dim = node_embeddings.size(-1)
                assert state.first_node_idx is not None
                assert state.last_node_idx is not None
                h_first = node_embeddings.gather(
                    dim=-2,
                    index=state.first_node_idx.unsqueeze(-1).expand(batch_size, 1, hidden_dim),
                ).squeeze(-2)
                h_last = node_embeddings.gather(
                    dim=-2,
                    index=state.last_node_idx.unsqueeze(-1).expand(batch_size, 1, hidden_dim),
                ).squeeze(-2)

            context_embedding = torch.cat([graph_embedding, h_last, h_first], dim=-1)
            context_embedding = self.context_projection(context_embedding)
        elif isinstance(state, CVRPConstructiveState):
            # CVRP context uses last selected node and remaining capacity.
            hidden_dim = node_embeddings.size(-1)
            assert state.last_node_idx is not None
            h_last = node_embeddings.gather(
                dim=-2,
                index=state.last_node_idx.unsqueeze(-1).expand(batch_size, 1, hidden_dim),
            ).squeeze(-2)
            context_embedding = torch.cat(
                [graph_embedding, h_last, state.remaining_capacity], dim=-1
            )
            context_embedding = self.context_projection(context_embedding)
        else:
            raise TypeError(f"Unsupported constructive state: {type(state)}")

        action_mask = state.action_mask

        # MHA: context node attends over all unmasked nodes
        context_embedding, _ = self.MHA(
            query=context_embedding.unsqueeze(1),
            key=node_embeddings,
            value=node_embeddings,
            key_padding_mask=action_mask,
        )
        context_embedding = context_embedding.squeeze(1)  # (batch, d_model)

        # Final single-head layer
        log_p = self.final_scorer(
            h_context=context_embedding, h_nodes=node_embeddings, mask=action_mask
        )
        return log_p
