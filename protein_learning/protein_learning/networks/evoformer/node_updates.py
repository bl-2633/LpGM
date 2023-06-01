"""Functions for performing Scalar Feature Attention Updates"""
import torch
from torch import nn, einsum, Tensor
from einops import rearrange
from protein_learning.networks.common.utils import exists, default
from typing import Optional
from protein_learning.networks.common.invariant.units import Residual, FeedForward, PreNorm

max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa


class NodeAttention(nn.Module):
    """Scalar Feature masked attention with pair bias"""

    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            edge_dim=None,
            include_bias=False,
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=include_bias)
        self.to_bias = nn.Linear(edge_dim, heads, bias=include_bias)
        self.to_out_node = nn.Linear(inner_dim, dim)
        self.edge_norm, self.node_norm = nn.LayerNorm(edge_dim), nn.LayerNorm(dim)

    def forward(self, nodes: Tensor, edges: Tensor, adj_mask: Tensor) -> Tensor:
        """Multi-head scalar Attention Layer

        :param nodes: scalar features of shape (b,n,d_s)
        :param edges: pair features of shape (b,n,n,d_e)
        :param adj_mask: boolean tensor of node adjacencies
        :return:
        """
        nodes, edges = self.node_norm(nodes), self.edge_norm(edges)
        h = self.heads
        q, k, v = self.to_qkv(nodes).chunk(3, dim=-1)
        b = self.to_bias(edges)
        q, k, v, b = map(lambda t: rearrange(t, 'b ... (h d) -> b h ... d', h=h), (q, k, v, b))
        node_logits = self._attn(q, k, v, b, adj_mask)
        node_logits = rearrange(node_logits, 'b h n d -> b n (h d)', h=h)
        return self.to_out_node(node_logits)

    def _attn(self, q, k, v, b, mask) -> Tensor:
        """Perform attention update"""
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale + b.squeeze(-1)
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim, dim=-1)
        # attention weightes sum of values
        return einsum('b h i j, b h j d -> b h i d', attn, v)


class NodeUpdateLayer(nn.Module):
    """Scalar Feature Update Layer
    Input scalar_feats
    (1) Pair Biased Attention (norm(scalar_feats))
    (2) Norm + Residual
    (3) Feed Forward
    (4) Residual
    """

    def __init__(self,
                 dim: int,
                 dim_head: int = 32,
                 heads: int = 8,
                 edge_dim: Optional[int] = None,
                 include_bias: bool = False,
                 use_rezero: bool = True,
                 ff_mult: int = 4,
                 dropout: float = 0.0,
                 ):
        super().__init__()
        self.attn = NodeAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            edge_dim=edge_dim,
            include_bias=include_bias
        )
        self.attn_residual = Residual(use_rezero=use_rezero)
        self.transition = PreNorm(dim, FeedForward(dim, dim_hidden=ff_mult * dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.transition_residual = Residual(use_rezero=use_rezero)

    def forward(self, nodes, edges, adj_mask: Optional[Tensor] = None) -> Tensor:
        """Perform Scalar Updates"""
        feats = self.attn(nodes=nodes, edges=edges, adj_mask=adj_mask)
        feats = self.attn_residual(feats, nodes)
        return self.transition_residual(self.dropout(self.transition(feats)), feats)
