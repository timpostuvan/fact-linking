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
            input_size=sent_dim + sent_dim,
            hidden_size=fc_dim,
            output_size=2,
            num_layers=n_fc_layers,
            dropout=dropout_prob_fc,
            layer_norm=True
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=2)

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

    def convert_to_probability(self, x):
        return (x + 1) / 2

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
        node_embeddings = torch.cat([sent_vecs_projected, self.concept_emb(concept_ids[:, 1:] - 1)], dim=1)    # (batch_size, n_node, dim_sent)
        num_nodes = node_embeddings.shape[1]

        sent_vecs_projected = sent_vecs_projected.repeat(1, num_nodes, 1)
        logits = self.convert_to_probability(self.cosine_similarity(sent_vecs_projected, node_embeddings))     # (batch_size, num_nodes)

        return logits, -1


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
