import torch


def build_encoder_input_indexes_from_sequence(
    sequences: torch.Tensor,  # (B, L)
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parse flat sequences into encoder inputs.

    Args:


    Returns:
        neighbours: (B, N, 2)
        route_index: (B, N)
    """
    B = sequences.shape[0]
    N = num_customers
    device = sequences.device

    neighbours = torch.zeros(B, N, 2, dtype=torch.long, device=device)
    tour_index = torch.zeros(B, N, dtype=torch.long, device=device)

    for b in range(B):
        seq = sequences[b]
        depot_positions = (seq == 0).nonzero(as_tuple=True)[0]
        for tour_idx in range(len(depot_positions) - 1):
            start = depot_positions[tour_idx].item()
            end = depot_positions[tour_idx + 1].item()
            tour = seq[start : end + 1]  # includes depot on both sides
            for pos in range(1, len(tour) - 1):
                customer = tour[pos].item()
                tour_index[b, customer - 1] = tour_idx
                neighbours[b, customer - 1] = torch.tensor(
                    [tour[pos - 1].item(), tour[pos + 1].item()]
                )

    return neighbours, tour_index


def build_encoder_input_indexes_from_routes(
    routes_batch: list[list[torch.Tensor]],
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        neighbours_index: (B, N, 2) [prev, next] node index per customer, 1-indexed
        tour_index: (B, N) local tour id per customer, 0-indexed
    """
    B = len(routes_batch)
    N = num_customers
    device = routes_batch[0][0].device

    neighbours = torch.zeros(B, N, 2, dtype=torch.long, device=device)
    tour_index = torch.zeros(B, N, dtype=torch.long, device=device)

    for b_idx, routes in enumerate(routes_batch):
        tour_with_depot = [[0, *route, 0] for route in routes]
        for tour_idx, tour in enumerate(tour_with_depot):
            for pos in range(1, len(tour) - 1):
                customer = tour[pos]
                tour_index[b_idx, customer - 1] = tour_idx
                neighbours[b_idx, customer - 1] = torch.tensor([tour[pos - 1], tour[pos + 1]])

    return neighbours, tour_index


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
