import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class RobertaTextEncoder(nn.Module):
    def __init__(self, encoder_name: str, from_checkpoint: str = None):
        super().__init__()
        
        self.module = AutoModel.from_pretrained(encoder_name, output_hidden_states=True)
        if from_checkpoint is not None:
            self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
        self.sent_dim = self.module.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids, layer_id=-1):
        outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]
        sent_vecs = self.module.pooler(hidden_states)
        return sent_vecs


class SentenceTextEncoder(nn.Module):
    def __init__(self, encoder_name: str, from_checkpoint: str = None):
        super().__init__()

        self.module = AutoModel.from_pretrained(encoder_name, output_hidden_states=True)
        if from_checkpoint is not None:
            self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
        self.sent_dim = self.module.config.hidden_size

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]      # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, token_type_ids, layer_id=-1):
        outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sent_vecs = self.mean_pooling(outputs, attention_mask)
        sent_vecs = F.normalize(sent_vecs, p=2, dim=-1)
        return sent_vecs
