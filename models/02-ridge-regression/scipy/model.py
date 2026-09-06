import logging

import numpy as np
from scipy.optimize import least_squares

log = logging.getLogger(__name__)


class RidgeRegressionLSQ:
    """Ridge fit via scipy.optimize.least_squares using the classic
    "augmented residuals" trick: appending sqrt(alpha) * weights to the
    residual vector reproduces the alpha * sum(weights^2) penalty once
    least_squares sums the squared residuals."""

    def __init__(self, n_features: int, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha
        self.params: np.ndarray | None = None  # [w_0..w_{k-1}, bias]

    def _residuals(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        log.debug(f"params={params}")

        # TODO(you): return the CONCATENATION of two blocks:
        #   1. the ordinary prediction residuals: (X @ weights + bias) - y,
        #      shape (n_samples,) -- same as the plain-OLS scipy runtime.
        #   2. sqrt(self.alpha) * weights, shape (n_features,) -- squaring
        #      and summing this block inside least_squares' objective gives
        #      exactly self.alpha * sum(weights**2), the ridge penalty.
        #      (weights = params[:-1], bias = params[-1]; the bias is
        #      intentionally left out of this second block -- it's never
        #      penalized.)
        # Use np.concatenate([block1, block2]).
        raise NotImplementedError("Implement the ridge-augmented residual function")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionLSQ":
        log.debug(f"X shape={X.shape}, y shape={y.shape}")
        x0 = np.zeros(self.n_features + 1)
        result = least_squares(self._residuals, x0, args=(X, y))
        self.params = result.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Call fit() before predict()")
        weights, bias = self.params[:-1], self.params[-1]
        return X @ weights + bias
