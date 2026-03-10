import torch
from torch_geometric.data import Data


def make_complete_graph(x: torch.Tensor, self_connections: bool = True) -> Data:
    """Create a fully-connected PyG Data object for a single instance."""
    n = x.size(0)
    idx = torch.arange(n)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    if not self_connections:
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
    edge_index = torch.stack([src, dst], dim=0).to(x.device)
    return Data(x=x, edge_index=edge_index)
