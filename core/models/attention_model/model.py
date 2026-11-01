import torch
import torch.nn as nn

from core.models.attention_model.decoder import Decoder
from core.models.attention_model.encoder import Encoder
from core.models.entropy import entropy
from core.models.rollout import Rollout
from core.problems.cvrp.constructive.env import CVRPConstructiveEnv
from core.problems.cvrp.features import CVRPInstanceFeatureExtractor
from core.problems.instance import BaseProblemInstance
from core.problems.tsp.constructive.env import TSPConstructiveEnv
from core.problems.tsp.features import TSPInstanceFeatureExtractor


class AttentionModel(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str = "tsp",
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()
        if problem_name not in {"tsp", "cvrp"}:
            raise ValueError(f"Unknown problem_name: {problem_name}")

        self.problem_name = problem_name

        match problem_name:
            case "cvrp":
                self.feature_extractor = CVRPInstanceFeatureExtractor()
                self.env = CVRPConstructiveEnv()
            case "tsp":
                self.feature_extractor = TSPInstanceFeatureExtractor()
                self.env = TSPConstructiveEnv()
            case _:
                raise ValueError(f"Unsupported problem_name: {problem_name}")

        self.encoder = Encoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.decoder = Decoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

    def forward(
        self,
        instance: BaseProblemInstance,
        *,
        greedy: bool = False,
        previous_actions: torch.Tensor | None = None,
    ) -> Rollout:
        return self.rollout(
            instance,
            greedy=greedy,
            previous_actions=previous_actions,
        )

    def rollout(
        self,
        instance: BaseProblemInstance,
        *,
        greedy: bool = False,
        previous_actions: torch.Tensor | None = None,
    ) -> Rollout:
        batch_size = instance.batch_size[0]
        n = instance.num_nodes

        features = self.feature_extractor(instance)
        h_nodes, h_graph = self.encoder(features)

        log_p_of_selected = []
        entropies, entropies_norm = [], []

        actions: list[torch.Tensor]

        if previous_actions is None:
            state = self.env.reset(instance)
            actions = []
        else:
            state = self.env.replay(instance, previous_actions)
            actions = list(previous_actions.unbind(-1))

        while not self.env.all_done(state):
            log_p = self.decoder(
                node_embeddings=h_nodes,
                graph_embedding=h_graph,
                state=state,
                action_mask=self.env.action_mask(state),
            )
            selected = log_p.argmax(-1) if greedy else log_p.exp().multinomial(1).squeeze(-1)
            log_p_of_selected.append(log_p.gather(-1, selected.unsqueeze(-1)).squeeze(-1))
            actions.append(selected)
            state = self.env.step(state, selected)

            step_entropy = entropy(log_p)
            entropies.append(step_entropy.entropy)
            entropies_norm.append(step_entropy.entropy_norm)

        actions_tensor = (
            torch.stack(actions, dim=-1)
            if actions
            else torch.zeros(batch_size, 0, dtype=torch.long, device=h_nodes.device)
        )
        log_p_tensor = (
            torch.stack(log_p_of_selected, dim=-1).sum(dim=-1)
            if log_p_of_selected
            else torch.zeros(batch_size, device=h_nodes.device)
        )
        entropy_tensor = (
            torch.stack(entropies, dim=-1).mean(dim=-1)
            if entropies
            else torch.zeros(batch_size, device=h_nodes.device)
        )
        entropy_norm_tensor = (
            torch.stack(entropies_norm, dim=-1).mean(dim=-1)
            if entropies_norm
            else torch.zeros(batch_size, device=h_nodes.device)
        )

        return Rollout.create(
            actions=actions_tensor,
            log_prob=log_p_tensor,
            entropy=entropy_tensor,
            entropy_norm=entropy_norm_tensor,
        )
