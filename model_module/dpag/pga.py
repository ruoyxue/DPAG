import torch.nn as nn


class PoseGraphAggregation(nn.Module):
    def __init__(self, N, node_dim, pga_node_dim, dropout_rate) -> None:
        super().__init__()
        self.linear_node = nn.Sequential(
            nn.Linear(node_dim * N, pga_node_dim),
            nn.LayerNorm(pga_node_dim)
        )

    def forward(self, node_feat):
        """ node_feat (B, T, N, D) temporal_variation (B, N, T, T, D) """
        b, t, n, d = node_feat.shape

        node_feat = node_feat.transpose(2, 3).reshape(b, t, -1)
        frame_feat = self.linear_node(node_feat)
        return frame_feat
