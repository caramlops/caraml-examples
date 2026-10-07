import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class PoissonRegressionModule(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_features))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log.debug(f"x shape={x.shape}")

        # TODO(you): implement eta = x @ self.weights + self.bias.
        #   - x has shape (batch, n_features)
        #   - self.weights has shape (n_features,)
        #   - return the raw linear predictor eta (= log of the predicted
        #     mean count), shape (batch,) -- do NOT apply exp() here.
        #     train.py's loss (nn.PoissonNLLLoss, log_input=True by
        #     default) expects log(mean) directly and applies exp()
        #     internally as part of a single, numerically stable loss
        #     computation -- the exact same "pass the pre-link-function
        #     value, let the loss apply the link" pattern as
        #     03-logistic-regression's CrossEntropyLoss expecting raw
        #     logits rather than softmax probabilities.
        raise NotImplementedError("Implement the matmul + add forward pass")
