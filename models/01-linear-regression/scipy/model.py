import logging

import numpy as np
from scipy import linalg

log = logging.getLogger(__name__)


class LinearRegressionOLS:
    """OLS fit via the closed-form normal equation, same math as the numpy
    runtime, but solved with scipy.linalg instead of numpy.linalg -- it lets
    you hand the solver a hint about the matrix's structure instead of
    running a generic solve."""

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionOLS":
        X_aug = np.c_[X, np.ones(X.shape[0])]  # add a factor for bias
        X_aug_t = X_aug.transpose()

        # (p + 1, p + 1), each side is p factors + 1 bias factor
        X_sq = X_aug_t @ X_aug
        log.debug(f"X_aug shape={X_aug.shape}, X_sq shape={X_sq.shape}")

        w_aug = linalg.solve(X_sq, X_aug_t @ y, assume_a="pos")

        self.weights = w_aug[:-1]
        self.bias = float(w_aug[-1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return X @ self.weights + self.bias
