import torch
import torch.nn.functional as F
from torch import nn


class BaseMessagePassing(nn.Module):
    def __init__(
        self,
        n_vertex_types: int,
        n_edge_types: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_size == output_size
        self.n_vertex_types = n_vertex_types
        self.n_edge_types = n_edge_types

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Embedding(self.n_vertex_types, hidden_size // 2)
        self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)

        self.edge_encoder = nn.Sequential(
            nn.Linear(n_edge_types + 1 + n_vertex_types * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def embed_edge(
        self,
        edge_index_ids,
        edge_type_ids,
        node_type_ids,
    ):
        # Prepare edge feature
        edge_vec = F.one_hot(edge_type_ids, self.n_edge_types + 1)  # [E, 39]
        self_edge_vec = torch.zeros(node_type_ids.size(0), self.n_edge_types + 1).to(edge_vec.device)
        self_edge_vec[:, self.n_edge_types] = 1
        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]

        head_type = node_type_ids[edge_index_ids[0]]  # [E,] #head=src
        tail_type = node_type_ids[edge_index_ids[1]]  # [E,] #tail=tgt
        head_vec = F.one_hot(head_type, self.n_vertex_types)  # [E,4]
        tail_vec = F.one_hot(tail_type, self.n_vertex_types)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = F.one_hot(node_type_ids, self.n_vertex_types)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]

        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]
        return edge_embeddings

    def forward(
        self,
        node_features,
        edge_index_ids,
        edge_type_ids,
        node_type,
        node_score
    ):
        batch_size, n_nodes, d_nodes = node_features.shape

        # Embed type
        node_type_emb = self.emb_node_type(node_type)
        node_type_emb = self.activation(node_type_emb)  # [batch_size, n_node, dim/2]

        # Embed score
        js = torch.arange(self.hidden_size // 2).unsqueeze(0).unsqueeze(0).float().to(node_type.device)  # [1,1,dim/2]
        js = torch.pow(1.1, js)  # [1,1,dim/2]
        node_score_encoding = torch.sin(js * node_score.unsqueeze(2))  # [batch_size, n_node, dim/2]
        node_score_emb = self.activation(self.emb_score(node_score_encoding))  # [batch_size, n_node, dim/2]

        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2)
        _node_feature_extra = _node_feature_extra.view(batch_size * n_nodes, -1).contiguous()

        # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _node_features = node_features.view(-1, d_nodes).contiguous()  # [b_size * n_node, d_node]
        _node_type = node_type.view(-1).contiguous()

        edge_embeddings = self.embed_edge(
            edge_index_ids=edge_index_ids,
            edge_type_ids=edge_type_ids,
            node_type_ids=_node_type
        )

        # Add self loops to edge_index
        loop_index = torch.arange(0, _node_features.size(0), dtype=torch.long, device=edge_index_ids.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index_ids = torch.cat([edge_index_ids, loop_index], dim=1)  # [2, E+N]

        _output_features = self.forward_gnn(
            x=_node_features,
            x_extra=_node_feature_extra,
            edge_embeddings=edge_embeddings,
            edge_index=edge_index_ids,
        )

        output_features = _output_features.view(batch_size, n_nodes, -1)  # [batch_size, n_node, dim]

        output = self.activation(self.Vh(node_features) + self.Vx(output_features))
        output = self.dropout(output)

        return output
