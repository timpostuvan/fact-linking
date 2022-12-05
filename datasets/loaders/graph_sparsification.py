from typing import Dict
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch import bernoulli



data_folder = "data"
pretrained_node_embeddings_folder = join(data_folder, "embeddings")
pretrained_node_embeddings_path = join(pretrained_node_embeddings_folder, "augmented_graph_embeddings.npy")


def load_resources():
    pretrained_node_embeddings = np.load(pretrained_node_embeddings_path)
    pretrained_node_embeddings = torch.tensor(pretrained_node_embeddings, dtype=torch.float)

    num_nodes, embedding_dim = pretrained_node_embeddings.shape[0], pretrained_node_embeddings.shape[1]
    node_embeddings = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)
    node_embeddings.weight.data.copy_(pretrained_node_embeddings)

    return node_embeddings


node_embeddings = load_resources()


def random_graph_sparsification(
    edge_index: torch.LongTensor,
    edge_types: torch.LongTensor,
    drop_p: float = 0.5
):
    num_edges = edge_index.shape[1]

    # drop edges uniformly at random with probability drop_p
    drop_probabilities = torch.ones(num_edges) * drop_p
    keep_edges = (bernoulli(drop_probabilities) == 0).bool()
    edge_index = edge_index[:, keep_edges]
    edge_types = edge_types[keep_edges]
    return edge_index, edge_types


def embedding_similarity_graph_sparsification(
    edge_index: torch.LongTensor,
    edge_types: torch.LongTensor,
    node2idx: Dict,
    similarity_threshold: float = 0.9
):
    idx2node = {idx:node for node, idx in node2idx.items()}

    _edge_index = edge_index.clone()
    for i in range(_edge_index.shape[0]):
        for j in range(_edge_index.shape[1]):
            _edge_index[i, j] = idx2node[_edge_index[i, j].item()]
    
    edge_embeddings = node_embeddings(_edge_index)
    edge_cosine_similarity = F.cosine_similarity(edge_embeddings[0], edge_embeddings[1])

    # keep edges that connect nodes with cosine similarity greater than threshold
    keep_edges = (edge_cosine_similarity > similarity_threshold).bool()
    edge_index = edge_index[:, keep_edges]
    edge_types = edge_types[keep_edges]
    return edge_index, edge_types
