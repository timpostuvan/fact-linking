import torch
from torch import nn
from torch_geometric.nn import GENConv

from models.gnn.base import BaseMessagePassing


class QAGEN(BaseMessagePassing):
    def __init__(
        self,
        n_gnn_layers: int,
        n_vertex_types: int,
        n_edge_types: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super(QAGEN, self).__init__(
            n_vertex_types=n_vertex_types,
            n_edge_types=n_edge_types,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
        )

        self.n_gnn_layers = n_gnn_layers
        self.gen = GENConv(
            in_channels=input_size,
            out_channels=output_size,
            num_layers=n_gnn_layers
        )
        self.projector = nn.Linear(2 * input_size, input_size)

    def forward_gnn(self, x, x_extra, edge_index, edge_embeddings):
        x = self.projector(torch.cat([x, x_extra], dim=1))
        return self.gen(x, edge_index, edge_embeddings)

