import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class RidgeRegressionModule(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_features))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same matmul + bias-add as plain linear regression -- ridge only
        # changes the loss (see l2_penalty below), not the forward pass.
        log.debug(f"x shape={x.shape}")
        return x @ self.weights + self.bias

    def l2_penalty(self, alpha: float) -> torch.Tensor:
        return alpha * sum(self.weights**2)
