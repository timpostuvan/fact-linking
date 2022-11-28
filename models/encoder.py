import torch
from torch import nn
import torch.nn.functional as F
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
        self.encoder_name = encoder_name

        self.module = AutoModel.from_pretrained(encoder_name, output_hidden_states=True)
        if from_checkpoint is not None:
            self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
        self.sent_dim = self.module.config.hidden_size

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, layer_id=-1):
        outputs = self.module(input_ids=input_ids, attention_mask=attention_mask)

        if "sentence-transformers" in self.encoder_name:
            all_hidden_states = outputs[0]
            sent_vecs = self.mean_pooling(outputs, attention_mask)
            sent_vecs = F.normalize(sent_vecs, p=2, dim=-1)
        else:
            all_hidden_states = outputs[-1]
            hidden_states = all_hidden_states[layer_id]
            sent_vecs = self.module.pooler(hidden_states)

        return sent_vecs, all_hidden_states
