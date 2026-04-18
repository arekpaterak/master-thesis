from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from core.models.attention_model.decoder import Decoder
from core.models.attention_model.encoder import Encoder
from core.models.attention_model.preprocessing import make_complete_graph
from core.models.entropy import entropy
from core.problems.cvrp import CVRPInstance
from core.problems.cvrp.features import CVRPInstanceFeatureExtractor
from core.problems.cvrp.solution import CVRPSolution
from core.problems.instance import BaseProblemInstance
from core.problems.tsp import TSPInstance
from core.problems.tsp.features import TSPInstanceFeatureExtractor
from core.problems.tsp.solution import TSPSolution
from core.states.constructive import CVRPConstructiveState, TSPConstructiveState


@dataclass(frozen=True)
class AMRollout:
    actions: torch.Tensor
    log_p: torch.Tensor
    entropy: float | None
    entropy_norm: float | None

    @property
    def tour(self) -> torch.Tensor:
        return self.actions

    @property
    def sequence(self) -> torch.Tensor:
        return self.actions

    def to_solution(self, instance: BaseProblemInstance):
        if isinstance(instance, TSPInstance):
            return TSPSolution.create(instance=instance, tour=self.actions)
        if isinstance(instance, CVRPInstance):
            return CVRPSolution.create(instance=instance, sequence=self.actions)
        raise TypeError(f"Unsupported instance type: {type(instance)}")


class AttentionModel(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str = "tsp",
        d_h: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        feature_extractor: Callable[[BaseProblemInstance], dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__()
        if problem_name not in {"tsp", "cvrp"}:
            raise ValueError(f"Unknown problem_name: {problem_name}")

        self.problem_name = problem_name
        self._feature_extractor_override = feature_extractor

        if feature_extractor is not None:
            self._feature_extractor_override = feature_extractor
        elif problem_name == "cvrp":
            self._feature_extractor_override = CVRPInstanceFeatureExtractor()
        elif problem_name == "tsp":
            self._feature_extractor_override = TSPInstanceFeatureExtractor()
        else:
            raise ValueError(f"Unsupported problem_name: {problem_name}")
        self.encoder = Encoder(
            problem_name=problem_name,
            d_h=d_h,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.decoder = Decoder(
            problem_name=problem_name,
            hidden_dim=d_h,
            num_heads=num_heads,
        )

    # TODO: Change to single step, add rollout to produce full sequence
    def forward(
        self,
        instance: BaseProblemInstance,
        *,
        greedy: bool = False,
        previous_actions: torch.Tensor | None = None,
    ) -> AMRollout:
        """
        Args:
            instance: (batch, n, d_x)
            greedy: if True, argmax; if False, sample
            previous_actions: (batch, k) optional pre-placed nodes. Assumes same number of pre-placed nodes across batch.
        Returns:
            RolloutResult
        """
        batch_size, n, _ = instance.locations.shape

        data = Batch.from_data_list(
            [make_complete_graph(instance.locations[b]) for b in range(batch_size)]
        )
        assert data["edge_index"].shape[0] == 2, (
            "Expected edge_index to have shape (2, total_edges)"
        )
        if self.problem_name == "cvrp" and not isinstance(instance, CVRPInstance):
            raise TypeError("AttentionModel(problem_name='cvrp') expects CVRPInstance input.")
        if self.problem_name == "tsp" and not isinstance(instance, TSPInstance):
            raise TypeError("AttentionModel(problem_name='tsp') expects TSPInstance input.")

        features = self._feature_extractor_override(instance)
        h_nodes, h_graph = self.encoder(
            edge_index=data["edge_index"],
            batch=data["batch"],
            **features,
        )
        h_nodes = h_nodes.view(batch_size, n, -1)

        log_p_of_selected = []
        entropies, entropies_norm = [], []

        actions: list[torch.Tensor]

        if self.problem_name == "tsp":
            if previous_actions is None:
                state = TSPConstructiveState.initial(
                    batch_size=batch_size,
                    num_nodes=n,
                    device=instance.locations.device,
                )
                actions = []
            else:
                state = TSPConstructiveState.from_partial(
                    partial_tour=previous_actions,
                    num_nodes=n,
                )
                actions = [previous_actions[:, i] for i in range(previous_actions.shape[1])]
        elif self.problem_name == "cvrp":
            state = CVRPConstructiveState.initial(
                demand=instance.demand,
                capacity=instance.capacity,
            )
            actions = []
        else:
            raise ValueError("Unknown problem type.")

        while not state.all_done():
            log_p = self.decoder(node_embeddings=h_nodes, graph_embedding=h_graph, state=state)
            selected = log_p.argmax(-1) if greedy else log_p.exp().multinomial(1).squeeze(-1)
            log_p_of_selected.append(log_p.gather(-1, selected.unsqueeze(-1)).squeeze(-1))
            actions.append(selected)
            state = state.transition(selected)

            step_entropy = entropy(log_p)
            entropies.append(step_entropy.entropy)
            entropies_norm.append(step_entropy.entropy_norm)

        mean_entropy = torch.stack(entropies, dim=-1).mean().item() if entropies else 0.0
        mean_entropy_norm = (
            torch.stack(entropies_norm, dim=-1).mean().item() if entropies_norm else 0.0
        )
        log_p_tensor = (
            torch.stack(log_p_of_selected, dim=-1)
            if log_p_of_selected
            else torch.zeros(batch_size, 0, device=h_nodes.device)
        )

        return AMRollout(
            actions=torch.stack(actions, dim=-1),
            log_p=log_p_tensor,
            entropy=mean_entropy,
            entropy_norm=mean_entropy_norm,
        )
