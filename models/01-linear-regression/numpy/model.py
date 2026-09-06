import logging

import numpy as np

log = logging.getLogger(__name__)


class LinearRegressionOLS:
    """Ordinary least squares, solved via the closed-form normal equation."""

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionOLS":
        log.debug(f"X shape={X.shape}, y shape={y.shape}")

        X_aug = np.c_[X, np.ones(X.shape[0])]
        X_aug_t = X_aug.transpose()
        X_sq = X_aug_t @ X_aug

        log.debug(f"X_aug shape={X_aug.shape}, X_aug_t shape={X_aug_t.shape}, X_sq shape={X_sq.shape}")

        w_aug = np.linalg.inv(X_sq) @ X_aug_t @ y

        log.debug(f"w_aug={w_aug}")

        self.weights = w_aug[:-1]
        self.bias = w_aug[-1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return X @ self.weights + self.bias
