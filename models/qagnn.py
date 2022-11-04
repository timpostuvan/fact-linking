from platform import node
from typing import Optional

import torch
from torch import nn

from datasets.batch_sample import BatchedSample
from models import TextEncoder
from models.gnn.gat import QAGAT
from models.gnn.gen import QAGEN
from models.layers import MultiheadAttPoolLayer, MLP, CustomizedEmbedding


class QAGNN(nn.Module):
    def __init__(
        self,
        sent_dim: int,
        gnn_name: str, n_gnn_layers: int, n_attn_head: int, dropout_prob_gnn: int,
        n_vertex_types: int, n_edge_types: int,
        fc_dim: int, n_fc_layers: int, dropout_prob_fc: int,
        n_concept: int, concept_dim: int, concept_in_dim: int, dropout_prob_emb: int,
        pretrained_concept_emb: Optional[torch.Tensor] = None, freeze_ent_emb: bool = True,
        init_range: float = 0.02,
    ):
        super().__init__()
        self.init_range = init_range

        self.concept_emb = CustomizedEmbedding(
            concept_num=n_concept,
            concept_out_dim=concept_dim,
            concept_in_dim=concept_in_dim,
            pretrained_concept_emb=pretrained_concept_emb,
            freeze_ent_emb=freeze_ent_emb
        )
        self.sent_projection = nn.Linear(sent_dim, concept_dim)
        self.activation = nn.GELU()

        if gnn_name == "gat":
            gnn_model = QAGAT
        elif gnn_name == "gen":
            gnn_model = QAGEN
        else:
            raise ValueError(f"Unknown gnn name: {gnn_name}")
        self.gnn = gnn_model(
            n_gnn_layers=n_gnn_layers,
            n_vertex_types=n_vertex_types,
            n_edge_types=n_edge_types,
            input_size=concept_dim,
            hidden_size=concept_dim,
            output_size=concept_dim,
            n_attn_head=n_attn_head,
            dropout=dropout_prob_gnn,
        )

        self.pooler = MultiheadAttPoolLayer(n_attn_head, sent_dim, concept_dim)

        self.fc = MLP(
            input_size=concept_dim + concept_dim,
            hidden_size=fc_dim,
            output_size=2,
            num_layers=n_fc_layers,
            dropout=dropout_prob_fc,
            layer_norm=True
        )

        self.dropout_e = nn.Dropout(dropout_prob_emb)
        self.dropout_fc = nn.Dropout(dropout_prob_fc)

        if init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        sent_vecs,
        concept_ids,
        node_type_ids,
        node_scores,
        adj_lengths,
        edge_index_ids,
        edge_type_ids,
    ):
        
        sent_vecs_projected = self.activation(self.sent_projection(sent_vecs)).unsqueeze(1)     # (batch_size, 1, dim_node)
        
        concept_vecs = self.concept_emb(concept_ids[:, 1:] - 1)     # (batch_size, n_node - 1, dim_node)

        gnn_input = self.dropout_e(torch.cat([sent_vecs_projected, concept_vecs], dim=1))   # (batch_size, n_node, dim_node)

        gnn_output = self.gnn(
            gnn_input,
            edge_index_ids,
            edge_type_ids,
            node_type_ids,
            node_scores
        )

        sent_vecs_after_gnn = gnn_output[:, 0]      # (batch_size, dim_node)

        # 1 means masked out
        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1)

        # pool over all KG nodes, but not QAGNN_contextnode
        mask = mask | (node_type_ids == 1)

        # a temporary solution to avoid zero node
        mask[mask.all(1), 0] = 0

        graph_vecs, pool_attn = self.pooler(sent_vecs, gnn_output, mask)

        node_embeddings = gnn_input     # (batch_size, n_node, dim_node)
        num_nodes = node_embeddings.shape[1]
        graph_vecs = graph_vecs.unsqueeze(1).repeat(1, num_nodes, 1)    # (batch_size, n_node, dim_node)
        sent_vecs = sent_vecs.unsqueeze(1).repeat(1, num_nodes, 1)      # (batch_size, n_node, dim_sent)
        sent_vecs_after_gnn = sent_vecs_after_gnn.unsqueeze(1).repeat(1, num_nodes, 1)  # (batch_size, n_node, dim_node)

        # when predicting relevance of a node, we consider: 
        # node embedding and embedding of QAGNN_contextnode
        concat = torch.cat([node_embeddings, sent_vecs_after_gnn], dim=-1)  
        concat = self.dropout_fc(concat)
        logits = self.fc(concat).transpose(1, 2)    # (batch_size, 2, num_nodes)
        return logits, pool_attn


class LM_QAGNN(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        gnn_name: str, n_gnn_layers: int, n_attn_head: int, dropout_prob_gnn: int,
        n_vertex_types: int, n_edge_types: int,
        fc_dim: int, n_fc_layers: int, dropout_prob_fc: int,
        n_concept: int, concept_dim: int,
        concept_in_dim: int, dropout_prob_emb: int,
        pretrained_concept_emb: Optional[torch.Tensor] = None, freeze_ent_emb: bool = True,
        init_range: float = 0.0,
    ):
        super().__init__()
        self.encoder = TextEncoder(encoder_name=encoder_name)
        self.decoder = QAGNN(
            gnn_name=gnn_name,
            n_gnn_layers=n_gnn_layers,
            n_vertex_types=n_vertex_types,
            n_edge_types=n_edge_types,
            sent_dim=self.encoder.sent_dim,
            n_concept=n_concept,
            concept_dim=concept_dim,
            concept_in_dim=concept_in_dim,
            n_attn_head=n_attn_head,
            fc_dim=fc_dim,
            n_fc_layers=n_fc_layers,
            dropout_prob_emb=dropout_prob_emb,
            dropout_prob_gnn=dropout_prob_gnn,
            dropout_prob_fc=dropout_prob_fc,
            pretrained_concept_emb=pretrained_concept_emb,
            freeze_ent_emb=freeze_ent_emb,
            init_range=init_range,
        )

    def forward(self, batch: BatchedSample, layer_id: int = -1):
        input_ids, attention_mask = batch.lm_inputs()
        sent_vecs, all_hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_id=layer_id
        )
        concept_ids, node_type_ids, node_scores, adj_lengths = batch.gnn_data()
        edge_index_ids, edge_type_ids = batch.adj()
        logits, attn = self.decoder(
            sent_vecs=sent_vecs,
            concept_ids=concept_ids,
            node_type_ids=node_type_ids,
            node_scores=node_scores,
            adj_lengths=adj_lengths,
            edge_index_ids=edge_index_ids,
            edge_type_ids=edge_type_ids
        )
        return logits, attn
