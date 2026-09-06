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

        # TODO(you): implement ridge's closed-form solve.
        #
        # X has shape (n_samples, n_features), y has shape (n_samples,).
        #   1. Augment X with a column of ones (shape (n_samples, n_features+1)),
        #      same as plain OLS.
        #   2. Build the penalty matrix: P = self.alpha * np.eye(n_features+1),
        #      then set P's LAST diagonal entry to 0. Ridge should never
        #      shrink the intercept -- only the feature weights.
        #   3. Solve: w_aug = (X_aug^T X_aug + P)^-1 X_aug^T y
        #      (this is exactly the OLS normal equation with `+ P` added --
        #      that addition is what keeps the matrix invertible even when
        #      X^T X is nearly singular from collinear columns).
        #   4. Split w_aug into self.weights (first n_features entries) and
        #      self.bias (last entry), as in OLS.
        raise NotImplementedError("Implement the ridge closed-form solver")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return X @ self.weights + self.bias
