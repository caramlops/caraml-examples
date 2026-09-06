import numpy as np


class LinearRegressionOLS:
    """Ordinary least squares, solved via the closed-form normal equation."""

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionOLS":
        # TODO(you): implement the OLS closed-form solve.
        #
        # X has shape (n_samples, n_features), y has shape (n_samples,).
        #   1. Augment X with a column of ones (shape (n_samples, n_features+1))
        #      so the bias is folded into a single weight vector.
        #   2. Solve the normal equation for the augmented weights `w_aug`:
        #         w_aug = (X_aug^T X_aug)^-1 X_aug^T y
        #      (np.linalg.inv + matmuls, or np.linalg.lstsq / np.linalg.solve
        #      for better numerical stability — any of these are fine).
        #   3. Split w_aug into self.weights (first n_features entries) and
        #      self.bias (last entry).
        raise NotImplementedError("Implement the OLS normal-equation solver")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return X @ self.weights + self.bias
