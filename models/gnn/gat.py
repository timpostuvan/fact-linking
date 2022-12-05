import math

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.utils import softmax
from torch_scatter import scatter

from models.gnn.base import BaseMessagePassing


class GAT(nn.Module):
    def __init__(
        self,
        n_gnn_layers: int,
        n_attn_head: int,
        input_size: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert output_size % n_attn_head == 0

        self.n_gnn_layers = n_gnn_layers
        self.gnn = nn.ModuleList([
            GATConv(
                in_channels=input_size,
                out_channels=output_size // n_attn_head,
                heads=n_attn_head,
                dropout=dropout
            )]
            +
            [GATConv(
                in_channels=output_size,
                out_channels=output_size // n_attn_head,
                heads=n_attn_head,
                dropout=dropout
            )
            for _ in range(n_gnn_layers - 1)]
        )

        self.dropouts = [nn.Dropout(dropout) for _ in range(n_gnn_layers)]
        self.activation = nn.GELU()

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.n_gnn_layers):
            x = self.gnn[i](x, edge_index, edge_attr)
            x = self.activation(x)
            x = self.dropouts[i](x)
        return x


class QAGAT(BaseMessagePassing):
    def __init__(
        self,
        n_gnn_layers: int,
        n_vertex_types: int,
        n_edge_types: int,
        n_attn_head: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super(QAGAT, self).__init__(
            n_vertex_types=n_vertex_types,
            n_edge_types=n_edge_types,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
        )

        self.n_gnn_layers = n_gnn_layers
        self.gat = nn.ModuleList([
            GATConvE(hidden_size, n_vertex_types, n_edge_types, n_attn_head, batch_norm=True)
            for _ in range(n_gnn_layers)
        ])

        self.dropouts = [nn.Dropout(dropout) for _ in range(n_gnn_layers)]

    def forward_gnn(self, x, x_extra, edge_index, edge_embeddings):
        for _ in range(self.n_gnn_layers):
            x = torch.cat([x, x_extra], dim=1)
            x = self.gat[_]((x, x), edge_index, edge_embeddings)
            x = self.activation(x)
            x = self.dropouts[_](x)
        return x


class GATConvE(MessagePassing):
    def __init__(self, emb_dim, n_ntype, n_etype, head_count=5, aggr="add", batch_norm: bool = True):
        super(GATConvE, self).__init__(aggr=aggr)

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype
        self.n_etype = n_etype

        # For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        if batch_norm:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, emb_dim)
            )
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, emb_dim)
            )

    def forward(self, x, edge_index, edge_embeddings, return_attention_weights=False):
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)
        # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)
        # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # cd n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]

        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]
