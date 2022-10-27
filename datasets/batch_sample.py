from collections import defaultdict
from typing import Sequence, Dict, Any

import torch


class BatchedSample:
    def __init__(self, batch: Sequence[Dict[str, Any]]):
        self.labels = []
        self.edge_index_ids, self.edge_type_ids = [], []
        self.encoder_data, self.decoder_data = defaultdict(list), defaultdict(list)

        self.batch_size = len(batch)

        for elem in batch:
            self.labels.append(elem["label"])
            self.edge_index_ids.append(elem["edge_index"])
            self.edge_type_ids.append(elem["edge_type"])

            for k, v in elem["encoder_data"].items():
                self.encoder_data[k].append(v)
            for k, v in elem["decoder_data"].items():
                self.decoder_data[k].append(v)

    def __len__(self):
        return len(self.labels)

    def to_tensors(self):
        for k in self.encoder_data:
            self.encoder_data[k] = torch.stack(self.encoder_data[k])
        for k in self.decoder_data:
            self.decoder_data[k] = torch.stack(self.decoder_data[k])
        self.labels = torch.stack(self.labels)
        
        n_examples, n_nodes = len(self.edge_index_ids), self.decoder_data["node_ids"].shape[1]
        self.edge_index_ids = [self.edge_index_ids[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        self.edge_index_ids = torch.cat(self.edge_index_ids, dim=1)
        self.edge_type_ids = torch.cat(self.edge_type_ids, dim=0)

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)

    def to_device(self, device: torch.device):
        if isinstance(self.labels, torch.Tensor):
            self.labels = self.labels.to(device)
        else:
            raise ValueError("self.labels are not yet torch.Tensor, run to_device method first")
        
        for k in self.encoder_data:
            if isinstance(self.encoder_data[k], torch.Tensor):
                self.encoder_data[k] = self.encoder_data[k].to(device)
            else:
                raise ValueError(f"self.encoder_data[{k}] are not yet torch.Tensor, run to_device method first")

        for k in self.decoder_data:
            if isinstance(self.decoder_data[k], torch.Tensor):
                self.decoder_data[k] = self.decoder_data[k].to(device)
            else:
                raise ValueError(f"self.decoder_data[{k}] are not yet torch.Tensor, run to_device method first")

        self.edge_index_ids = self._to_device(self.edge_index_ids, device)
        self.edge_type_ids = self._to_device(self.edge_type_ids, device)

    def lm_inputs(self):
        return self.encoder_data.values()

    def adj(self):
        return self.edge_index_ids, self.edge_type_ids

    def gnn_data(self):
        return self.decoder_data.values()
