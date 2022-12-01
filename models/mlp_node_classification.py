from typing import Optional

import torch
from torch import nn

from datasets.batch_sample import BatchedSample
from models import RobertaTextEncoder, SentenceTextEncoder
from models.layers import CustomizedEmbedding, MLP


class MLPDecoder(nn.Module):
    def __init__(
        self,
        sent_dim: int,
        fc_dim: int,
        n_fc_layers: int,
        dropout_prob_fc: int,
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
        self.dropout_e = nn.Dropout(dropout_prob_emb)

        self.sent_projection = nn.Linear(sent_dim, concept_dim)
        self.activation = nn.GELU()
    
        self.fc = MLP(
            input_size=concept_dim + concept_dim,
            hidden_size=fc_dim,
            output_size=2,
            num_layers=n_fc_layers,
            dropout=dropout_prob_fc,
            layer_norm=True
        )
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
        # (batch_size, 1, dim_node)
        sent_vecs_projected = self.activation(self.sent_projection(sent_vecs)).unsqueeze(1)     

        # (batch_size, n_node - 1, dim_node)
        concept_vecs = self.concept_emb(concept_ids[:, 1:] - 1)

        # (batch_size, n_node, dim_node)
        node_embeddings = self.dropout_e(torch.cat([sent_vecs_projected, concept_vecs], dim=1))
        num_nodes = node_embeddings.shape[1]
        
        # (batch_size, n_node, dim_node)
        sent_vecs_projected = sent_vecs_projected.repeat(1, num_nodes, 1)
        
        # when predicting relevance of a node, we consider: 
        # context LM embedding and static node embedding 
        # (batch_size, n_node, 2 * dim_node)
        concat = torch.cat([sent_vecs_projected, node_embeddings], dim=-1)
        concat = self.dropout_fc(concat)

        # (batch_size, 2, num_nodes)
        logits = self.fc(concat).transpose(1, 2)

        return logits


class MLPNodeClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        fc_dim: int,
        n_fc_layers: int,
        dropout_prob_fc: int,
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

        self.decoder = MLPDecoder(
            sent_dim=self.encoder.sent_dim,
            fc_dim=fc_dim,
            n_fc_layers=n_fc_layers,
            dropout_prob_fc=dropout_prob_fc,
            n_concept=n_concept,
            concept_dim=concept_dim,
            concept_in_dim=concept_in_dim,
            dropout_prob_emb=dropout_prob_emb,
            pretrained_concept_emb=pretrained_concept_emb,
            freeze_ent_emb=freeze_ent_emb,
            init_range=init_range,
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
