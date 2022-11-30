import pickle
import json
from typing import Dict
from os.path import join

import torch
from torch.utils.data import Dataset

from .loaders.text_loader import load_text_input_tensors
from .loaders.graph_loader import load_sparse_adj_data_with_contextnode


data_folder = "data"
processed_dataset_folder = join(data_folder, "processed-comfact")
fact_grounding_path = join(processed_dataset_folder, "fact2idx.json")


def load_resources():
    with open(fact_grounding_path, "r") as f:
        _fact2idx  = json.load(f)

    _idx2fact = {idx:fact for fact, idx in _fact2idx.items()}
    return _idx2fact


def load_node_classification_data(path: str):
    with open(path, 'rb') as f:
        samples = pickle.load(f)

    text_data, graph_data, labels = [], [], []
    for sample in samples:
        text_data.append(" ".join(sample["text_context_list"]))
        graph_data.append(dict(
            nodes=sample["nodes"],
            node_types=sample["node_types"],
            edges=sample["edges"],
            edge_types=sample["edge_types"],
            idx2score=sample["idx2score"]
        ))
        labels.append(sample["labels"])

    return text_data, graph_data, labels


def load_graph_classification_data(path: str, idx2fact: Dict):
    with open(path, 'rb') as f:
        samples = pickle.load(f)

    text_data, graph_data, labels = [], [], []
    for sample in samples:
        text = " ".join(sample["text_context_list"])
        num_labeled_nodes = len(sample["labels"])
        for i in range(num_labeled_nodes):
            node_idx = sample["nodes"][i]
            fact_text = idx2fact[node_idx]
            text_data.append(text + " " + fact_text)
            graph_data.append(dict(
                nodes=sample["nodes"],
                node_types=sample["node_types"],
                edges=sample["edges"],
                edge_types=sample["edge_types"],
                idx2score=sample["idx2score"]
            ))
            labels.append(sample["labels"][i])

    return text_data, graph_data, 


class ComFactDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        classification_task_type: str,
        batch_size: int,
        model_name: str,
        max_node_num: int = 200,
        max_seq_length: int = 128
    ):
        super().__init__()
        self.batch_size = batch_size

        if classification_task_type == "node":
            text_data, graph_data, labels = load_node_classification_data(dataset_path)
            self.num_samples = len(labels)

            label_lengths = torch.tensor([len(label_set) for label_set in labels], dtype=torch.long)
            self.labels = torch.full((self.num_samples, max_node_num), -1, dtype=torch.long)        # default -1: not a valid label anymore
            for idx, label_set in enumerate(labels):
                self.labels[idx, 1:label_lengths[idx] + 1] = torch.tensor(label_set, dtype=torch.long)

        elif classification_task_type == "graph":
            idx2fact = load_resources()

            text_data, graph_data, labels = load_graph_classification_data(dataset_path, idx2fact)
            self.num_samples = len(text_data)
            self.labels = torch.tensor(labels)
        else:
            raise ValueError(f"Unknown classification task type {classification_task_type}")

        encoder_data = load_text_input_tensors(
            texts=text_data,
            model_name=model_name,
            max_seq_length=max_seq_length
        )

        self.input_ids = encoder_data["input_ids"]
        self.attention_mask = encoder_data["attention_mask"]
        self.token_type_ids = encoder_data["token_type_ids"]

        *decoder_data, adj_data = load_sparse_adj_data_with_contextnode(
            graph_data=graph_data,
            max_node_num=max_node_num
        )

        self.node_ids, self.node_type_ids, self.node_scores, self.adj_lengths = decoder_data
        self.edge_index, self.edge_type = adj_data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        output = {
            "label": self.labels[idx],
            "encoder_data": {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "token_type_ids": self.token_type_ids[idx]
            },
            "decoder_data": {
                "node_ids": self.node_ids[idx],
                "node_type_ids": self.node_type_ids[idx],
                "node_scores": self.node_scores[idx],
                "adj_lengths": self.adj_lengths[idx]
            },
            "edge_index": self.edge_index[idx],
            "edge_type": self.edge_type[idx],
        }
        return output
