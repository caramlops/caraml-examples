import logging

import torch

log = logging.getLogger(__name__)


class RidgeRegressionRaw:
    """Ridge fit via raw torch.Tensor autograd -- no nn.Module, no
    nn.Parameter, no torch.optim. Same idea as
    01-linear-regression/torch/model_raw.py; the only difference is fit()
    adds the L2 penalty into the loss before calling _step()."""

    def __init__(self, n_features: int, alpha: float = 1.0, lr: float = 0.05, n_iters: int = 200):
        self.alpha = alpha
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

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "RidgeRegressionRaw":
        for i in range(self.n_iters):
            preds = self.predict(X)
            mse = torch.mean((preds - y) ** 2)
            penalty = self.alpha * torch.sum(self.weights**2)
            loss = mse + penalty
            self._step(loss)
            if i % 50 == 0:
                log.info(f"epoch {i}: train loss (MSE + penalty) {loss.item():.4f}")
        return self
