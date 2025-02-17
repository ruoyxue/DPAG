import torch
import torch.nn as nn

from einops import repeat
from .sparse_ops import SparseOps


class PoseGraphRefinementUnit(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout_rate, scale) -> None:
        super().__init__()
        self.linear_node = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim * scale),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(node_dim * scale, node_dim)
        )

        self.linear_sp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * scale),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(edge_dim * scale, edge_dim)
        )

        self.linear_tv = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * scale),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(edge_dim * scale, edge_dim)
        )

        self.linear_sp2node = nn.Sequential(
            nn.Linear(edge_dim, node_dim)
        )

        self.linear_tv2node = nn.Sequential(
            nn.Linear(edge_dim, node_dim)
        )

    def forward(self, node, sp, tv, padding_mask, local_mask):
        """ node (B, T, N, D) sp (B, T, N, N, D) tv (B, N, T, T, D) or (B, N, M, D) padding_mask (B, 1, T) local_mask (T, T) """
        B, T, N = node.shape[:3]

        sp_attn = torch.softmax(sp, dim=-2) * sp
        sp_node = self.linear_sp2node(sp_attn.sum(-2))  # (B, T, N, D)
        if padding_mask is not None:
            sp_node = sp_node.masked_fill(padding_mask.squeeze(1)[..., None, None] == 0, 0)  # (B, T, N, D)

        local_mask = local_mask.unsqueeze(-1)  # (T, T, 1)

        if padding_mask is not None:
            padding_mask = repeat(padding_mask.squeeze(1), 'b t -> b n q1 t q2', n=N, q1=1, q2=1)  # (B, N, 1, T, 1)
            scores_tv = tv.masked_fill(padding_mask == 0, 1e-28).masked_fill(local_mask == 0, 1e-28)
            tv_attn = torch.softmax(scores_tv, dim=-2) * tv
            tv_attn = tv_attn.masked_fill(padding_mask == 0, 0).masked_fill(local_mask == 0, 0)
        else:
            scores_tv = tv.masked_fill(local_mask == 0, 1e-28)
            tv_attn = torch.softmax(scores_tv, dim=-2) * tv
            tv_attn = tv_attn.masked_fill(local_mask == 0, 0)

        tv_node = self.linear_tv2node(tv_attn.sum(-2)).transpose(1, 2)  # (B, T, N, D)

        node = self.linear_node(node + sp_node + tv_node)
        sp = self.linear_sp(sp)

        sparse_tv = SparseOps(tv, local_mask.squeeze(-1))
        dense_tv = self.linear_tv(sparse_tv.reshape_to_dense())
        sparse_tv.update_sparse(dense_tv)
        tv = sparse_tv.sparse.to_dense()

        return node, sp, tv
