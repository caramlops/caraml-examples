import logging

import numpy as np
from scipy import linalg

log = logging.getLogger(__name__)


class RidgeRegression:
    """Ridge fit via the closed-form normal equation, same math as the numpy
    runtime, but solved with scipy.linalg instead of numpy.linalg."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        log.debug(f"X shape={X.shape}, y shape={y.shape}")

        # TODO(you): implement ridge's closed-form solve using
        # scipy.linalg.solve(...) instead of numpy.linalg.solve.
        #
        # X has shape (n_samples, n_features), y has shape (n_samples,).
        #   1. Augment X with a column of ones (shape (n_samples, n_features+1)),
        #      same as plain OLS.
        #   2. Build the penalty matrix: P = self.alpha * np.eye(n_features+1),
        #      then set P's LAST diagonal entry to 0 -- ridge never shrinks
        #      the intercept, only the feature weights.
        #   3. Solve: (X_aug^T X_aug + P) @ w_aug = X_aug^T y via
        #      scipy.linalg.solve. Note X_aug^T X_aug + P is symmetric
        #      positive-definite (P only adds to the diagonal), so you can
        #      pass assume_a="pos" here too, same as the plain-OLS scipy
        #      runtime.
        #   4. Split w_aug into self.weights (first n_features entries) and
        #      self.bias (last entry), as in OLS.
        num_samples, num_features = X.shape

        X_aug = np.c_[X, np.ones(num_samples)]  # add a factor for bias
        X_aug_t = X_aug.transpose()

        penalty = self.alpha * np.eye(num_features + 1)
        penalty[-1] = 0.0

        # (p + 1, p + 1), each side is p factors + 1 bias factor
        X_sq = X_aug_t @ X_aug
        log.debug(f"X_aug shape={X_aug.shape}, X_sq shape={X_sq.shape}")

        w_aug = linalg.solve(X_sq + penalty, X_aug_t @ y, assume_a="pos")

        self.weights = w_aug[:-1]
        self.bias = float(w_aug[-1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return X @ self.weights + self.bias
