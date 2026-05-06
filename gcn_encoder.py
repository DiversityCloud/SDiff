# -*- coding: utf-8 -*-
"""GCN-based social condition encoder for SDiff."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def normalize_edge_index(edge_index) -> torch.Tensor:
    """Convert edge_index to a LongTensor with shape [2, num_edges]."""
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.dim() != 2:
        raise ValueError(f"edge_index must be 2-D, got shape={tuple(edge_index.shape)}")
    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got shape={tuple(edge_index.shape)}")
    return edge_index.contiguous()


class FullGraphGCNUserEncoder(nn.Module):
    """Full-graph GCN user encoder.

    Dropout is controlled by the `dropout` argument and is applied after each
    GCN layer. In main.py, this value is exposed as `--dropout`.
    """

    def __init__(self, num_users: int, embed_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.convs = nn.ModuleList([GCNConv(embed_dim, embed_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.user_embedding.weight
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.norm(x)


class GCNConditionalEncoder(nn.Module):
    """Return GCN-based social condition vectors for target users."""

    def __init__(self, gcn_encoder: FullGraphGCNUserEncoder, edge_index):
        super().__init__()
        if edge_index is None:
            raise ValueError("edge_index cannot be None for GCNConditionalEncoder.")
        self.gcn_encoder = gcn_encoder
        self.register_buffer("edge_index", normalize_edge_index(edge_index), persistent=False)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        all_user_emb = self.gcn_encoder(self.edge_index)
        return all_user_emb[user_ids]
