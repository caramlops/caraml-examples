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
        log.debug(f"x shape={x.shape}")

        # TODO(you): implement y = x @ weights + bias.
        #   - x has shape (batch, n_features)
        #   - self.weights has shape (n_features,)
        #   - matrix-multiply x by self.weights to get a (batch,) tensor of
        #     raw scores, then add self.bias (broadcasts over the batch dim).
        #
        # This runs once per epoch, so a stray log.debug() here shows up a
        # lot at CARAML_LOGLEVEL=DEBUG -- that's fine, it's silent otherwise.
        raise NotImplementedError("Implement the matmul + add forward pass")
