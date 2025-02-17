import torch
import torch.nn as nn

from .attention import SpatialAttention, TemporalAttention
from .pgru import PoseGraphRefinementUnit
from .pga import PoseGraphAggregation
from model_module.components.transformer.timesformer.rotary import AxialRotaryEmbedding, RotaryEmbedding
from .sparse_ops import create_local_mask


NORM = torch.nn.LayerNorm

class DPAEncoder(nn.Module):
    """ dynamic pose alignment encoder """
    def __init__(self, idim, node_dim, edge_dim, heads, layers, dropout_rate, attn_drop_rate, \
                 pga_node_dim, pgru_scale, macaron_style, tv_local_width) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        self.tv_local_width = tv_local_width
        
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(idim, node_dim)
        )

        dim_head = node_dim // heads
        self.temporal_rot_emb = RotaryEmbedding(dim_head)
        self.spatial_rot_emb = AxialRotaryEmbedding(dim_head)

        self.encoders = nn.ModuleList(
            [
                DPAModule(
                    node_dim=node_dim, 
                    edge_dim=edge_dim, 
                    pgru_scale=pgru_scale, 
                    heads=heads, 
                    dropout_rate=dropout_rate, 
                    attn_drop_rate=attn_drop_rate,
                    macaron_style=macaron_style,
                    tv_local_width=tv_local_width
                )
                for _ in range(layers)
            ]
        )

        self.after_norm = nn.LayerNorm(node_dim)
        
        self.pga = PoseGraphAggregation(4, node_dim, pga_node_dim, dropout_rate)
    
    def forward(self, node, padding_mask=None):
        """
        Args:
            node: (B, T, N, D)
            padding_mask: (B, 1, T)
        """
        B, T, N, D = node.shape
        
        if padding_mask is None:
            padding_mask = torch.ones((B, 1, T)).to(node.device)

        padding_mask = padding_mask.to(node.dtype)
        node = self.embed(node)
        
        assert self.tv_local_width % 2 == 1
        sp = torch.zeros(B, T, N, N, self.edge_dim, dtype=node.dtype).to(node.device)
        tv_local_mask = create_local_mask(T, self.tv_local_width)  # (T, T)
        tv = torch.zeros(B, N, T, T, self.edge_dim, dtype=node.dtype).to(node.device)

        temporal_pe = self.temporal_rot_emb(T, device = node.device)
        spatial_pe = self.spatial_rot_emb(int(N ** 0.5), int(N ** 0.5), device = node.device)
        tv_local_mask = tv_local_mask.to(node.device)
        
        for i in range(len(self.encoders)):
            node, sp, tv, padding_mask = self.encoders[i](node, sp, tv, padding_mask, tv_local_mask, temporal_pe, spatial_pe)

        node = self.after_norm(node)

        frame_feat = self.pga(node)

        return frame_feat


class DPAModule(nn.Module):
    """ dynamic pose alignment module """
    def __init__(self, node_dim, edge_dim, pgru_scale, heads, dropout_rate, attn_drop_rate, macaron_style, tv_local_width) -> None:
        super().__init__()
        self.macaron_style = macaron_style
        
        self.tv_local_width = tv_local_width

        self.scale = 1.
        if macaron_style:
            self.psru1 = PoseGraphRefinementUnit(
                node_dim=node_dim,
                edge_dim=edge_dim,
                dropout_rate=dropout_rate,
                scale=pgru_scale
            )
            
            self.norm_psru1_node = NORM(node_dim)
            self.norm_psru1_sp = NORM(edge_dim)
            self.norm_psru1_tv = NORM(edge_dim)
            self.scale = 0.5
        
        self.spatial_attn = SpatialAttention(
            node_dim=node_dim,
            edge_dim=edge_dim,
            heads=heads,
            att_dropout_rate=attn_drop_rate
        )
        
        self.temporal_attn = TemporalAttention(
            node_dim=node_dim,
            edge_dim=edge_dim,
            heads=heads,
            attn_drop_rate=attn_drop_rate,
            tv_local_width=tv_local_width
        )
        
        self.norm_spatial_node = NORM(node_dim)
        self.norm_spatial_sp = NORM(edge_dim)
        self.norm_temporal_node = NORM(node_dim)
        self.norm_temporal_tv = NORM(edge_dim)

        self.psru2 = PoseGraphRefinementUnit(
            node_dim=node_dim,
            edge_dim=edge_dim,
            dropout_rate=dropout_rate,
            scale=pgru_scale
        )
        
        self.norm_psru2_node = NORM(node_dim)
        self.norm_psru2_sp = NORM(edge_dim)
        self.norm_psru2_tv = NORM(edge_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node, sp, tv, padding_mask, local_mask, temporal_pe, spatial_pe, vis=False):
        if self.macaron_style:
            # psru1
            up_node = self.norm_psru1_node(node)
            up_sp = self.norm_psru1_sp(sp)
            up_tv = self.norm_psru1_tv(tv)
            up_node, up_sp, up_tv = self.psru1(up_node, up_sp, up_tv, padding_mask, local_mask)
            node = node + self.dropout(up_node) * self.scale
            sp = sp + self.dropout(up_sp) * self.scale
            tv = tv + self.dropout(up_tv) * self.scale
        
        # sptial attention
        up_node = self.norm_spatial_node(node)
        up_sp = self.norm_spatial_sp(sp)
        up_node, up_sp = self.spatial_attn(up_node, up_sp, padding_mask, spatial_pe, vis)
        node = node + self.dropout(up_node)
        sp = sp + self.dropout(up_sp)

        # temporal attention
        up_node = self.norm_temporal_node(node)
        up_tv = self.norm_temporal_tv(tv)
        up_node, up_tv = self.temporal_attn(up_node, up_tv, padding_mask, local_mask, temporal_pe, vis)
        node = node + self.dropout(up_node)
        tv = tv + self.dropout(up_tv)

        # psru2
        up_node = self.norm_psru2_node(node)
        up_sp = self.norm_psru2_sp(sp)
        up_tv = self.norm_psru2_tv(tv)
        up_node, up_sp, up_tv = self.psru2(up_node, up_sp, up_tv, padding_mask, local_mask)
        node = node + self.dropout(up_node) * self.scale
        sp = sp + self.dropout(up_sp) * self.scale
        tv = tv + self.dropout(up_tv) * self.scale
       
        return node, sp, tv, padding_mask
