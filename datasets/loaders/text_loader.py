from transformers import AutoTokenizer
from typing import List


def load_text_input_tensors(texts: List[str], model_name: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_features = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        return_token_type_ids=True,
        return_special_tokens_mask=True
    )

    return text_features
