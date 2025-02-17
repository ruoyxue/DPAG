import torch
import torch.nn as nn
import math
from einops import rearrange, repeat

from model_module.components.transformer.timesformer.rotary import apply_rot_emb


class SpatialAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, heads, att_dropout_rate) -> None:
        super().__init__()
        assert node_dim % heads == 0
        self.d_k = node_dim // heads
        self.h = heads
        self.clip_min, self.clip_max = [-5, 5]

        self.linear_qkv = nn.Linear(node_dim, node_dim * 3)
        self.linear_g = nn.Linear(edge_dim, heads)
        self.linear_update = nn.Linear(self.h, edge_dim)
        self.linear_out_node = nn.Linear(node_dim, node_dim)
        self.attn_dropout = nn.Dropout(att_dropout_rate)

    def forward_qkvg(self, node, sp, pe=None):
        b, t, n, d1 = node.shape
        d2 = sp.shape[-1]
        node = node.view(-1, n, d1)
        sp = sp.view(-1, n, n, d2)

        qkv = self.linear_qkv(node)
        q, k ,v = rearrange(qkv, 'b n (h d1 x) -> b h n d1 x', d1=self.d_k, x=3).unbind(-1)  # q k v (B * T, h, N, d_k)

        if pe is not None:
            q, k = apply_rot_emb(q, k, pe)
        
        g = self.linear_g(sp).permute(0, 3, 1, 2)

        return q, k, v, g

    def forward_attention(self, q, k, v, g, mask, vis):
        """ q k v (b, h, t, d)  g (b, h, t, t) mask (b, 1, t)"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + g
        scores = scores.clamp(self.clip_min, self.clip_max)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # (b, 1, 1, t)
            scores = scores.masked_fill(mask == 0, 1e-28)
            attn = torch.softmax(scores, dim=-1)
            attn = attn.masked_fill(mask == 0, 0)
        else:
            attn = torch.softmax(scores, dim=-1)  # (b, h, t, t)

        p_attn = self.attn_dropout(attn)

        x = torch.matmul(p_attn, v)  # (b, h, t, d)
        
        return x, scores

    def forward(self, node, sp, mask=None, pe=None, vis=False):
        """ node feat (B, T, N, D) sp (B, T, N, N, D) mask (B, 1, T)"""
        B, T, N, D = node.shape
        
        mask = mask.view(-1)  # (B * T)
        mask = repeat(mask, 'b -> b n1 n2', n1=N, n2=N)  # (b, N, N)
        
        q, k, v, g = self.forward_qkvg(node, sp, pe)

        x, scores = self.forward_attention(q, k, v, g, mask, vis)  # x (b, h, n, d) scores (b, h, n, n) b = B * T

        x = rearrange(x, '(b t) h n d1 -> b t n (h d1)', t = T)
        x = self.linear_out_node(x)

        sp = rearrange(scores, '(b t) h n1 n2 -> b t n1 n2 h', b = B)
        sp = self.linear_update(sp)   
        
        return x, sp


class TemporalAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, heads, attn_drop_rate, tv_local_width) -> None:
        super().__init__()
        assert node_dim % heads == 0
        self.d_k = node_dim // heads
        self.h = heads
        self.clip_min, self.clip_max = [-5, 5]

        self.linear_qkv = nn.Linear(node_dim, node_dim * 3)
        self.linear_g = nn.Linear(edge_dim, heads)
        self.linear_update = nn.Linear(self.h, edge_dim)
        self.linear_out_node = nn.Linear(node_dim, node_dim)
        self.attn_dropout = nn.Dropout(attn_drop_rate)

        self.lwidth = tv_local_width
    
    def forward_qkvg(self, node, tv, pe=None):
        b, t, n, d1 = node.shape
        d2 = tv.shape[-1]
        node = rearrange(node, 'b t n d -> (b n) t d')

        qkv = self.linear_qkv(node)
        q, k ,v = rearrange(qkv, 'b t (h d1 x) -> b h t d1 x', d1=self.d_k, x=3).unbind(-1)  # q k v (B * N, h, T, d_k)

        tv = tv.view(-1, t, t, d2)
        g = self.linear_g(tv).permute(0, 3, 1, 2)

        if pe is not None:
            q, k = apply_rot_emb(q, k, pe)
 
        return q, k, v, g

    def forward_attention(self, q, k, v, g, mask, local_mask, vis=False):
        """ q k v (b, h, t, d)  g (b, h, t, t) mask (b, 1, t)"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B * N, h, T, T)
        scores = scores + g * local_mask
        scores = scores.clamp(self.clip_min, self.clip_max) 

        if mask is not None:
            mask = mask.unsqueeze(1)  # (b, 1, 1, t)
            scores = scores.masked_fill(mask == 0, 1e-28)
            attn = torch.softmax(scores, dim=-1)
            attn = attn.masked_fill(mask == 0, 0)
        else:
            attn = torch.softmax(scores, dim=-1)  # (b, h, t, t)

        p_attn = self.attn_dropout(attn)

        x = torch.matmul(p_attn, v)  # (b, h, t, d)

        return x, scores

    def forward(self, node, tv, mask=None, local_mask=None, pe=None, vis=False):
        """ node (B, T, N, D) tv (B, N, T, T, D) mask (B, 1, T) """
        B, T, N, D = node.shape

        q, k, v, g = self.forward_qkvg(node, tv, pe)

        mask = repeat(mask, 'b m t -> (b n) m t', n = N)

        x, scores = self.forward_attention(q, k, v, g, mask, local_mask, vis)  # x (b, h, t, d) scores (b, h, t, t) b = B * N
        
        x = rearrange(x, '(b n) h t d1 -> b t n (h d1)', n = N)
        x = self.linear_out_node(x)

        scores = rearrange(scores, '(b n) h t1 t2 -> b n t1 t2 h', n = N)
        tv = self.linear_update(scores) 

        return x, tv
