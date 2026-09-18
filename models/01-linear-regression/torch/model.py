import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class LinearRegressionModule(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_features))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weights) + self.bias
