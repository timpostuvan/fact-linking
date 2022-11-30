from torch import nn

from datasets.batch_sample import BatchedSample
from models import RobertaTextEncoder, SentenceTextEncoder
from models.layers import MLP


class MLPHead(nn.Module):
    def __init__(
        self,
        sent_dim: int,
        fc_dim: int,
        n_fc_layers: int,
        dropout_prob_fc: int
    ):
        super().__init__()
        self.fc = MLP(
            input_size=sent_dim,
            hidden_size=fc_dim,
            output_size=2,
            num_layers=n_fc_layers,
            dropout=dropout_prob_fc,
            layer_norm=True
        )
        self.dropout_fc = nn.Dropout(dropout_prob_fc)

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
        # (batch_size, dim_sent)
        sent_vecs = self.dropout_fc(sent_vecs)
        # (batch_size, 2)
        logits = self.fc(sent_vecs)
        return logits


class LMGraphClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        fc_dim: int,
        n_fc_layers: int,
        dropout_prob_fc: int
    ):
        super().__init__()

        if "roberta" in encoder_name:
            self.encoder = RobertaTextEncoder(encoder_name=encoder_name)
        elif "sentence-transformers" in encoder_name:
            self.encoder = SentenceTextEncoder(encoder_name=encoder_name)
        else:
            raise ValueError(f"Unknown encoder name {encoder_name}")

        self.decoder = MLPHead(
            sent_dim=self.encoder.sent_dim,
            fc_dim=fc_dim,
            n_fc_layers=n_fc_layers,
            dropout_prob_fc=dropout_prob_fc
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
