import torch

from core.problems.cvrp.solution import CVRPSolution
from core.problems.features import FeatureExtractor


class CVRPFeatureExtractor(FeatureExtractor[CVRPSolution]):
    """
    Standard feature extractor for the default CVRP NDS Model.
    Extracts the full problem instance features and relations.
    """

    def __call__(self, solution: CVRPSolution) -> dict[str, torch.Tensor]:
        return {
            "depot_features": solution.instance.depot_location,
            "customer_features": solution.instance.customer_features,
            "neighbours_index": solution.neighbours_per_customer(),
            "tour_index": solution.route_per_customer(),
        }


class RepairCVRPFeatureExtractor(FeatureExtractor[CVRPSolution]):
    """
    Feature extractor for the CVRP repair model.
    Extracts only the nodes that are free to be reinserted (degree < 2).
    """

    def __call__(self, solution: CVRPSolution) -> dict[str, torch.Tensor]:
        b_size = solution.batch_size
        device = solution.device

        # 1. State of the solution and degree
        degree = solution.node_degree()

        # In CVRP, depot is always required. Customers with degree < 2 are free.
        relevant_customer_mask = degree < 2

        # 2. Subset dimensions
        max_k = int(relevant_customer_mask.sum(dim=-1).max().item())

        # Select indices of relevant nodes (top-k over mask)
        top_k = relevant_customer_mask.float().topk(max_k, dim=-1)
        subset_indices = top_k.indices
        valid_subset_mask = top_k.values.bool()

        # 3. Calculate Custom CVRP features: Fulfilled Demand & State
        # Calculate fulfilled demand per route
        route_ids = solution.route_per_customer()
        demand_per_route = solution.demand_per_route()

        # Map back to customers
        # If route_id is -1, demand is 0.0
        valid_route = route_ids != -1
        customer_fulfilled_demand = torch.where(
            valid_route,
            demand_per_route.gather(dim=-1, index=route_ids.clamp(min=0)),
            torch.tensor(0.0, device=demand_per_route.device),
        )

        # 4. Gather customer features
        extended_customer_features = torch.cat(
            [
                solution.instance.customer_features,
                customer_fulfilled_demand.unsqueeze(-1),
                degree.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        subset_customer_features = extended_customer_features.gather(
            dim=-2, index=subset_indices.unsqueeze(-1).expand(*b_size, max_k, 5)
        )
        subset_customer_features = subset_customer_features.masked_fill(
            ~valid_subset_mask.unsqueeze(-1), 0.0
        )

        # 5. Depot features
        depot_features = torch.cat(
            [
                solution.instance.depot_location,
                torch.zeros((*b_size, 1, 1), device=device),
                torch.full((*b_size, 1, 1), -1.0, device=device),
                torch.full((*b_size, 1, 1), -1.0, device=device),
            ],
            dim=-1,
        )

        return {
            "depot_features": depot_features,
            "customer_features": subset_customer_features,
            "padding_mask": ~valid_subset_mask,
            "original_indices": subset_indices,
        }
