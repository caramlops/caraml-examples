import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class RidgeRegressionLayer(nn.Module):
    """Same forward pass and L2 penalty as model.py, but built from
    nn.Linear instead of hand-declared nn.Parameters."""

    def __init__(self, n_features: int):
        super().__init__()

        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log.debug(f"x shape={x.shape}")
        return self.linear(x).squeeze(-1)

    def l2_penalty(self, alpha: float) -> torch.Tensor:
        return alpha * torch.sum(self.linear.weight**2)
