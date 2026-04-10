import torch

from core.problems.features import FeatureExtractor
from core.problems.tsp.solution import TSPSolution


class TSPFeatureExtractor(FeatureExtractor[TSPSolution]):
    """
    Standard feature extractor for the TSP NDS Model.
    Extracts the full problem instance features and topology.
    """

    def __call__(self, solution: TSPSolution) -> dict[str, torch.Tensor]:
        return {
            "node_features": solution.instance.locations,
            "neighbours_index": solution.neighbours_index(),
        }


class RepairTSPFeatureExtractor(FeatureExtractor[TSPSolution]):
    """
    Feature extractor for the TSP repair model.
    Extracts only the nodes that are free to be reinserted (degree < 2).
    """

    def __call__(self, solution: TSPSolution) -> dict[str, torch.Tensor]:
        B = solution.batch_size

        # 1. Calculate degree for each node
        degree = solution.node_degree()

        # 2. Identify relevant nodes (degree < 2)
        relevant_mask = degree < 2

        # 3. Determine max subset size in the batch
        k = int(relevant_mask.sum(dim=-1).max().item())  # max subset size

        # 4. Subset nodes using top-k trick to push irrelevant nodes to the end
        top_k = relevant_mask.float().topk(k, dim=-1)
        subset_original_indices = top_k.indices
        valid_subset_mask = top_k.values.bool()

        # 5. Gather features
        locations = solution.instance.locations.gather(
            dim=-2, index=subset_original_indices.unsqueeze(-1).expand(*B, k, 2)
        )
        locations = locations.masked_fill(~valid_subset_mask.unsqueeze(-1), 0.0)

        state = degree.gather(dim=-1, index=subset_original_indices).float()
        state = state.masked_fill(~valid_subset_mask, -1.0)

        node_features = torch.cat([locations, state.unsqueeze(-1)], dim=-1)

        return {
            "node_features": node_features,
            "padding_mask": ~valid_subset_mask,
            "original_indices": subset_original_indices,
        }
