import logging

import torch

log = logging.getLogger(__name__)


class LinearRegressionRaw:
    """OLS fit via raw torch.Tensor autograd -- no nn.Module, no
    nn.Parameter, no torch.optim. This is what nn.Module and
    optimizer.step() are normally hiding: gradients computed by
    .backward() walking the autograd graph torch built while you computed
    predictions and loss, and parameters updated by hand."""

    def __init__(self, n_features: int, lr: float = 0.05, n_iters: int = 200):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = torch.zeros(n_features, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights + self.bias

    def _step(self, loss: torch.Tensor) -> None:
        log.debug(f"weights={self.weights.detach()}, bias={
                  self.bias.detach()}")

        loss.backward()

        with torch.no_grad():
            self.weights -= self.lr * self.weights.grad
            self.bias -= self.lr * self.bias.grad

        self.weights.grad.zero_()
        self.bias.grad.zero_()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "LinearRegressionRaw":
        for i in range(self.n_iters):
            preds = self.predict(X)
            loss = torch.mean((preds - y) ** 2)
            self._step(loss)
            if i % 50 == 0:
                log.info(f"epoch {i}: train MSE {loss.item():.4f}")
        return self
