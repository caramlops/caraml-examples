import numpy as np
from scipy.optimize import least_squares


class LinearRegressionLSQ:
    """OLS fit via scipy.optimize.least_squares rather than a closed-form
    solve — the same objective, minimized numerically."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.params: np.ndarray | None = None  # [w_0..w_{k-1}, bias]

    def _residuals(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # TODO(you): return the residual vector (predictions - y) for the
        # current parameter guess `params`.
        #   - params[:-1] are the weights (shape (n_features,))
        #   - params[-1] is the bias (scalar)
        #   - X has shape (n_samples, n_features), y has shape (n_samples,)
        #   - predictions = X @ weights + bias, residuals = predictions - y
        raise NotImplementedError("Implement the residual function for least_squares")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionLSQ":
        x0 = np.zeros(self.n_features + 1)
        result = least_squares(self._residuals, x0, args=(X, y))
        self.params = result.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Call fit() before predict()")
        weights, bias = self.params[:-1], self.params[-1]
        return X @ weights + bias
