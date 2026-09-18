import logging

import numpy as np

log = logging.getLogger(__name__)


class RidgeRegressionGD:
    """Ridge fit via imperative batch gradient descent -- the manual-loop
    counterpart to model.py's closed-form solve. Same idea as the plain
    linear-regression numpy_gd runtime, plus the L2 penalty's own gradient
    term."""

    def __init__(self, n_features: int, alpha: float = 1.0, lr: float = 0.1, n_iters: int = 500):
        self.alpha = alpha
        self.lr = lr
        self.n_iters = n_iters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def _step(self, X: np.ndarray, residuals: np.ndarray) -> None:
        n_samples = X.shape[0]

        grad_weights = (2.0 / n_samples) * X.T @ residuals + 2 * self.alpha * self.weights
        grad_bias = (2.0 / n_samples) * residuals.sum()
        self.weights -= self.lr * grad_weights
        self.bias -= self.lr * grad_bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionGD":
        for i in range(self.n_iters):
            residuals = self.predict(X) - y
            self._step(X, residuals)
            if i % 100 == 0:
                penalty = self.alpha * np.sum(self.weights**2)
                log.debug(f"iter={i} mse={np.mean(
                    residuals**2):.4f} penalty={penalty:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
