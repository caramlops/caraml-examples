import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class LogisticRegressionModule(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_features, n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log.debug(f"x shape={x.shape}")

        # TODO(you): implement logits = x @ self.weights + self.bias.
        #   - x has shape (batch, n_features)
        #   - self.weights has shape (n_features, n_classes)
        #   - return raw logits, shape (batch, n_classes) -- do NOT apply
        #     softmax here. train.py's loss (nn.CrossEntropyLoss) expects
        #     raw logits and applies log-softmax internally, fused with
        #     the loss computation for numerical stability. Applying
        #     softmax yourself first and feeding probabilities to
        #     CrossEntropyLoss would silently apply softmax twice and
        #     produce a broken loss with no error raised.
        raise NotImplementedError("Implement the matmul + add forward pass")
