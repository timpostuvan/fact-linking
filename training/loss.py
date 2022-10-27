from omegaconf import DictConfig
from torch import nn


def get_loss(config: DictConfig, ignore_index: int = -100):
    if config.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss {config.loss}")

    def compute_loss(logits, labels):
        if config.loss == 'cross_entropy':
             loss = loss_func(logits, labels)
        else:
            raise ValueError(f"Unknown loss {config.loss}")
        return loss

    return compute_loss
