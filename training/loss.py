from typing import Optional
from omegaconf import DictConfig
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELossIgnoreIndex(nn.BCELoss):
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean', ignore_index: int = -100) -> None:
        super(BCELossIgnoreIndex, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = (target != self.ignore_index)
        input, target = input[mask], target[mask].float()
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class FocalLoss(nn.Module):
    """
    Focal loss implemented as a PyTorch module.
    [Original paper](https://arxiv.org/pdf/1708.02002.pdf).
    """

    def __init__(self, gamma: float, ignore_index: int, reduction='none'):
        """
        :param gamma: What value of Gamma to use. Value of 0 corresponds to Cross entropy.
        :param reduction: Reduction to be done on top of datapoint-level losses.
        """
        super().__init__()

        assert reduction in ['none', 'sum', 'mean']

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input_logits: torch.Tensor, targets: torch.Tensor):
        ce_loss = torch.nn.functional.cross_entropy(input_logits, targets, reduction='none', ignore_index=self.ignore_index)
        input_probs_for_target = torch.exp(-ce_loss)
        loss = (1 - input_probs_for_target) ** self.gamma * ce_loss

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def get_loss(config: DictConfig, ignore_index: int = -100):
    if config.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
    elif config.loss == 'binary_cross_entropy':
        loss_func = BCELossIgnoreIndex(reduction='mean', ignore_index=ignore_index)
    elif config.loss == 'focal_loss':
        loss_func = FocalLoss(gamma=2, reduction='mean', ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss {config.loss}")

    def compute_loss(logits, labels):
        if config.loss in ['cross_entropy', 'binary_cross_entropy', 'focal_loss']:
             loss = loss_func(logits, labels)
        else:
            raise ValueError(f"Unknown loss {config.loss}")
        return loss

    return compute_loss
