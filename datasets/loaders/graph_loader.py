import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict


def load_adj_data(
        graph_data: List[Dict],
        max_node_num: int
    ):

    n_samples = len(graph_data)
    edge_index_list, edge_type_list = [], []
    adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
    node_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
    node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)  # default 2: padding nodes
    node_scores = torch.zeros((n_samples, max_node_num), dtype=torch.float)     # dummy values for compatibility
    adj_lengths_ori = adj_lengths.clone()

    for idx, graph in tqdm(enumerate(graph_data), total=n_samples):
        nodes = graph["nodes"]      # node types shouldn't be considered
        edges, edge_types = graph["edges"], graph["edge_types"]

        num_nodes = min(len(nodes), max_node_num)
        # this is the final number of nodes excluding PAD
        adj_lengths_ori[idx] = len(nodes)
        adj_lengths[idx] = num_nodes

        # prepare nodes
        nodes = torch.tensor(nodes[:num_nodes])

        # prepare node types
        node_type_ids[idx, :num_nodes] = 1     # actual nodes

        # prepare original edges, keep only the ones that are not pruned
        node2idx = {v.item(): idx for idx, v in enumerate(nodes)}
        edge_index = torch.stack([torch.LongTensor([node2idx[u], node2idx[v]]) for u, v in edges 
            if u in node2idx and v in node2idx], dim=-1)
        edge_types = torch.tensor([edge_types[idx] for idx, (u, v) in enumerate(edges) 
            if u in node2idx and v in node2idx], dtype=torch.long)

        assert edge_index.shape[1] == edge_types.shape[0]
        
        edge_index_list.append(edge_index)  # each entry is [2, E]
        edge_type_list.append(edge_types)    # each entry is [E, ]


    ori_adj_mean = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    print('| original_adjacency_len: mu {:.2f} sigma {:.2f} | adjacency_len: {:.2f} |'.format(
        ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()
    ) + ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()))

    # node_ids: (n_samples, max_node_num)
    # node_type_ids: (n_samples, max_node_num)
    # node_scores: (n_samples, max_node_num)
    # adj_lengths: (n_samples,)
    # edge_index: list of size (n_samples), where each entry is tensor[2, E]
    # edge_type: list of size (n_samples,), where each entry is tensor[E, ]
    return node_ids, node_type_ids, node_scores, adj_lengths, (edge_index_list, edge_type_list)


def load_adj_data_with_contextnode(
        graph_data: List[Dict],
        max_node_num: int
    ):

    n_samples = len(graph_data)
    edge_index_list, edge_type_list = [], []
    adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
    node_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
    node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)  # default 2: padding nodes
    node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
    adj_lengths_ori = adj_lengths.clone()

    for idx, graph in tqdm(enumerate(graph_data), total=n_samples):
        nodes = graph["nodes"]      # node types shouldn't be considered
        edges, edge_types = graph["edges"], graph["edge_types"]
        idx2score = graph["idx2score"]

        num_nodes = min(len(nodes), max_node_num - 1) + 1
        # this is the final number of nodes including QAGNN_contextnode but excluding PAD
        adj_lengths_ori[idx] = len(nodes)
        adj_lengths[idx] = num_nodes

        # prepare nodes
        nodes = torch.tensor(nodes[:num_nodes - 1])
        # to accomodate QAGNN_contextnode, original node_ids are incremented by 1
        node_ids[idx, 1:num_nodes] = nodes + 1
        node_ids[idx, 0] = 0    # this is the node_id for QAGNN_contextnode

        # prepare node scores
        if idx2score is not None:
            for j in range(num_nodes):
                _node_id = int(node_ids[idx, j]) - 1
                assert _node_id in idx2score
                node_scores[idx, j, 0] = torch.tensor(idx2score[_node_id])

        # prepare node types
        node_type_ids[idx, 0] = 0               # QAGNN_contextnode
        node_type_ids[idx, 1:num_nodes] = 1     # actual nodes

        # prepare original edges, keep only the ones that are not pruned
        node2idx = {v.item(): idx for idx, v in enumerate(nodes)}
        edge_index = torch.stack([torch.LongTensor([node2idx[u], node2idx[v]]) for u, v in edges 
            if u in node2idx and v in node2idx], dim=-1)
        edge_types = torch.tensor([edge_types[idx] for idx, (u, v) in enumerate(edges) 
            if u in node2idx and v in node2idx], dtype=torch.long)

        assert edge_index.shape[1] == edge_types.shape[0]

        # add edges to QAGNN_contextnode
        edge_index = edge_index + 1     # increment node ids
        edge_types = edge_types + 1     # increment edge type ids

        # new edges have to be bidirectional, the original ones already are
        qagnn_context_node = torch.zeros(nodes.shape[0], dtype=torch.long)
        original_nodes = torch.arange(1, nodes.shape[0] + 1, dtype=torch.long)
        additional_edges = torch.stack(
            [torch.cat([qagnn_context_node, original_nodes]),
            torch.cat([original_nodes, qagnn_context_node])]
        )
        additional_edge_types = torch.zeros(additional_edges.shape[1], dtype=torch.long)

        # merge edges and edge types
        edge_index = torch.cat([edge_index, additional_edges], dim=-1)
        edge_types = torch.cat([edge_types, additional_edge_types], dim=-1)
        
        edge_index_list.append(edge_index)  # each entry is [2, E]
        edge_type_list.append(edge_types)    # each entry is [E, ]


    ori_adj_mean = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    print('| original_adjacency_len: mu {:.2f} sigma {:.2f} | adjacency_len: {:.2f} |'.format(
        ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()
    ) + ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()))

    # normalize node score
    # masked positions have value 0
    _mask = (torch.arange(node_scores.shape[1]) < adj_lengths.unsqueeze(1)).float() # [n_samples, max_node_num]

    node_scores = -node_scores
    node_scores = node_scores - node_scores[:, 0:1, :]  # [n_samples, max_node_num, 1]
    node_scores = node_scores.squeeze(-1)   # [n_samples, max_node_num]
    node_scores = node_scores * _mask
    mean_norm = (torch.abs(node_scores)).sum(dim=1) / adj_lengths   # [n_samples,]
    node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05)    # [n_samples, max_node_num]

    # node_ids: (n_samples, max_node_num)
    # node_type_ids: (n_samples, max_node_num)
    # node_scores: (n_samples, max_node_num)
    # adj_lengths: (n_samples,)
    # edge_index: list of size (n_samples), where each entry is tensor[2, E]
    # edge_type: list of size (n_samples,), where each entry is tensor[E, ]
    return node_ids, node_type_ids, node_scores, adj_lengths, (edge_index_list, edge_type_list)
