import logging

import numpy as np

log = logging.getLogger(__name__)


class RidgeRegression:
    """Ridge regression: OLS with an L2 penalty on the weights, solved via
    the closed-form normal equation with a shrinkage term added."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        log.debug(f"X shape={X.shape}, y shape={y.shape}")

        num_samples, num_features = X.shape

        X_aug = np.c_[X, np.ones(num_samples)]  # add a factor for bias
        X_aug_t = X_aug.T
        X_sq = X_aug_t @ X_aug

        penalty = self.alpha * np.eye(num_features + 1)
        penalty[-1] = 0.0  # set the last one zero so the bias isn't shrunk

        # solve does LU decomposition
        # d-RSS / d-beta = 0 => (X_sq + alpha @ I) @ beta = X_aug_t @ y
        w_aug = np.linalg.solve(X_sq + penalty, X_aug_t @ y)

        self.weights = w_aug[:-1]
        self.bias = float(w_aug[-1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return X @ self.weights + self.bias
