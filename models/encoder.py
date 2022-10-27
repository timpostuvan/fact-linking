from torch import nn
from transformers import AutoModel
from transformers import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
)

MODEL_CLASS_TO_NAME = {
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
}

MODEL_NAME_TO_CLASS = {
    model_name: model_class
    for model_class, model_name_list in MODEL_CLASS_TO_NAME.items()
    for model_name in model_name_list
}

# Add SapBERT configuration
model_name = 'prajjwal1/bert-tiny'
MODEL_NAME_TO_CLASS[model_name] = 'bert'

model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_TO_CLASS[model_name] = 'bert'


class TextEncoder(nn.Module):
    def __init__(self, encoder_name: str, output_token_states: bool = False, from_checkpoint: str = None):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[encoder_name]
        self.output_token_states = output_token_states
        assert not self.output_token_states or self.model_type in ('bert', 'roberta', 'albert')

        self.module = AutoModel.from_pretrained(encoder_name, output_hidden_states=True)
        if from_checkpoint is not None:
            self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
        self.sent_dim = self.module.config.hidden_size

    def forward(self, input_ids, attention_mask, layer_id=-1):
        outputs = self.module(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        if self.output_token_states:
            return hidden_states
        sent_vecs = self.module.pooler(hidden_states)
        return sent_vecs, all_hidden_states
