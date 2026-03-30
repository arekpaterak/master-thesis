import torch

from core.problems.cvrp import CVRPSolution


def build_encoder_input_indexes(
    solution: CVRPSolution,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parse flat sequences into encoder inputs.

    Args:
        solution: CVRPSolution with sequence of shape (B, S)

    Returns:
        neighbour_index: (B, N, 2)
        route_index: (B, N)
    """
    sequence = solution.sequence
    B = sequence.shape[0]
    N = solution.instance.num_customers
    device = sequence.device

    neighbour_index = -1 * torch.ones(B, N, 2, dtype=torch.long, device=device)
    route_index = -1 * torch.ones(B, N, dtype=torch.long, device=device)

    for b_idx in range(B):
        seq = sequence[b_idx]
        depot_positions = (seq == 0).nonzero(as_tuple=True)[0]
        for route_idx in range(len(depot_positions) - 1):
            start = depot_positions[route_idx].item()
            end = depot_positions[route_idx + 1].item()
            route = seq[start : end + 1]  # include depot on both sides
            for pos in range(1, len(route) - 1):
                customer = route[pos].item()
                route_index[b_idx, customer - 1] = route_idx
                neighbour_index[b_idx, customer - 1] = torch.tensor(
                    [route[pos - 1].item(), route[pos + 1].item()]
                )

    return neighbour_index, route_index


# def build_encoder_input_indexes_from_routes(
#     routes_batch: list[list[torch.Tensor]],
#     num_customers: int,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns:
#         neighbours_index: (B, N, 2) [prev, next] node index per customer, 1-indexed
#         tour_index: (B, N) tour id per customer, 0-indexed
#     """
#     B = len(routes_batch)
#     N = num_customers
#     device = routes_batch[0][0].device
#
#     neighbours = torch.zeros(B, N, 2, dtype=torch.long, device=device)
#     tour_index = torch.zeros(B, N, dtype=torch.long, device=device)
#
#     for b_idx, routes in enumerate(routes_batch):
#         tour_with_depot = [[0, *route, 0] for route in routes]
#         for tour_idx, tour in enumerate(tour_with_depot):
#             for pos in range(1, len(tour) - 1):
#                 customer = tour[pos]
#                 tour_index[b_idx, customer - 1] = tour_idx
#                 neighbours[b_idx, customer - 1] = torch.tensor([tour[pos - 1], tour[pos + 1]])
#
#     return neighbours, tour_index


# def native_scatter_mean(
#     src: torch.Tensor,
#     index: torch.Tensor,
#     dim: int = 0,
#     dim_size: int | None = None
# ) -> torch.Tensor:
#     if dim_size is None:
#         dim_size = int(index.max().item()) + 1
#
#     out_shape = list(src.shape)
#     out_shape[dim] = dim_size
#     out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
#
#     if index.ndim < src.ndim:
#         view_shape = [1] * src.ndim
#         view_shape[dim] = index.shape[dim]
#         expanded_index = index.view(view_shape).expand_as(src)
#     else:
#         expanded_index = index
#
#     return out.scatter_reduce(
#         dim=dim,
#         index=expanded_index,
#         src=src,
#         reduce="mean",
#         include_self=False
#     )
