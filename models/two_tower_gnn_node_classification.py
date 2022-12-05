from typing import Optional

import torch
from torch import nn

from datasets.batch_sample import BatchedSample
from models import RobertaTextEncoder, SentenceTextEncoder
from models.gnn.gat import GAT
from models.layers import CustomizedEmbedding


class TwoTowerGNN(nn.Module):
    def __init__(
        self,
        sent_dim: int,
        n_gnn_layers: int,
        n_attn_head: int,
        dropout_prob_gnn: int,
        n_concept: int,
        concept_dim: int,
        concept_in_dim: int,
        dropout_prob_emb: int,
        pretrained_concept_emb: Optional[torch.Tensor] = None,
        freeze_ent_emb: bool = True,
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

        self.gnn = GAT(
            n_gnn_layers=n_gnn_layers,
            n_attn_head=n_attn_head,
            input_size=concept_dim,
            output_size=concept_dim,
            dropout=dropout_prob_gnn,
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=2)

        self.dropout_e = nn.Dropout(dropout_prob_emb)

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
        # (batch_size, 1, dim_node)
        sent_vecs_projected = self.activation(self.sent_projection(sent_vecs)).unsqueeze(1)
   
        # (batch_size, n_node, dim_node)
        gnn_input = self.dropout_e(self.concept_emb(concept_ids))
        batch_size, num_nodes, concept_dim = gnn_input.shape

        gnn_output = self.gnn(
            gnn_input.reshape(-1, concept_dim),
            edge_index_ids,
            edge_type_ids
        )
        gnn_output = gnn_output.reshape(batch_size, num_nodes, concept_dim)

        # (batch_size, n_node, dim_node)
        node_embeddings = gnn_output

        # (batch_size, n_node, dim_node)
        sent_vecs_projected = sent_vecs_projected.repeat(1, num_nodes, 1)

        # (batch_size, num_nodes)
        logits = self.convert_to_probability(self.cosine_similarity(sent_vecs_projected, node_embeddings)) 
        return logits


class TwoTowerGNNNodeClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        n_gnn_layers: int,
        n_attn_head: int,
        dropout_prob_gnn: int,
        n_concept: int,
        concept_dim: int,
        concept_in_dim: int,
        dropout_prob_emb: int,
        pretrained_concept_emb: Optional[torch.Tensor] = None,
        freeze_ent_emb: bool = True,
        init_range: float = 0.0,
    ):
        super().__init__()

        if "roberta" in encoder_name:
            self.encoder = RobertaTextEncoder(encoder_name=encoder_name)
        elif "sentence-transformers" in encoder_name:
            self.encoder = SentenceTextEncoder(encoder_name=encoder_name)
        else:
            raise ValueError(f"Unknown encoder name {encoder_name}")

        self.decoder = TwoTowerGNN(
            n_gnn_layers=n_gnn_layers,
            sent_dim=self.encoder.sent_dim,
            n_concept=n_concept,
            concept_dim=concept_dim,
            concept_in_dim=concept_in_dim,
            n_attn_head=n_attn_head,
            dropout_prob_emb=dropout_prob_emb,
            dropout_prob_gnn=dropout_prob_gnn,
            pretrained_concept_emb=pretrained_concept_emb,
            freeze_ent_emb=freeze_ent_emb,
            init_range=init_range
        )

    def forward(self, batch: BatchedSample, layer_id: int = -1):
        input_ids, attention_mask, token_type_ids = batch.lm_inputs()
        sent_vecs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            layer_id=layer_id
        )
        concept_ids, node_type_ids, node_scores, adj_lengths = batch.gnn_data()
        edge_index_ids, edge_type_ids = batch.adj()
        logits = self.decoder(
            sent_vecs=sent_vecs,
            concept_ids=concept_ids,
            node_type_ids=node_type_ids,
            node_scores=node_scores,
            adj_lengths=adj_lengths,
            edge_index_ids=edge_index_ids,
            edge_type_ids=edge_type_ids
        )

        return logits
