import logging

import numpy as np

log = logging.getLogger(__name__)


class LinearRegressionOLS:
    """Ordinary least squares, solved via the closed-form normal equation."""

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionOLS":
        X_aug = np.c_[X, np.ones(X.shape[0])]  # add a factor for bias
        X_aug_t = X_aug.T

        # (p + 1, p + 1), each side is p factors + 1 bias factor
        X_sq = X_aug_t @ X_aug

        log.debug(f"X_aug={X_aug.shape} X_sq={
                  X_sq.shape}, cond={np.linalg.cond(X_sq):.2e}")

        # solve does LU decomposition
        # d-RSS / d-beta = 0 => X_sq @ beta = X_aug_t @ y
        w_aug = np.linalg.solve(X_sq, X_aug_t @ y)

        self.weights = w_aug[:-1]
        self.bias = float(w_aug[-1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return X @ self.weights + self.bias
