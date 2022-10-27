from torch import nn

from datasets.batch_sample import BatchedSample
from models.layers import MLP


class TextClassifier(nn.Module):
    def __init__(self, text_encoder, fc_dim=128, n_fc_layer=3, p_fc=0.1):
        super().__init__()
        self.encoder = text_encoder

        self.activation = nn.GELU()

        self.decoder = MLP(text_encoder.config.hidden_size, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_fc = nn.Dropout(p_fc)

    def forward(self, batch: BatchedSample, layer_id: int = -1):
        input_ids, attention_mask, token_type_ids, output_mask = batch.lm_inputs()
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        concat = self.dropout_fc(outputs.pooler_output)
        logits = self.decoder(concat)
        logits = logits.view(batch.batch_size, batch.num_classes)
        return logits, None
