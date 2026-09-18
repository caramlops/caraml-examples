import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class LinearRegressionLayer(nn.Module):
    """Same OLS forward pass as model.py, but built from nn.Linear instead
    of hand-declared nn.Parameters -- nn.Linear owns its own weight matrix
    and bias as trainable parameters internally, the way Keras's Dense does
    in the tensorflow runtime."""

    def __init__(self, n_features: int):
        super().__init__()

        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
