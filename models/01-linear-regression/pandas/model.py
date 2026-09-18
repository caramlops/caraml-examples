import logging

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


class LinearRegressionOLS:
    """OLS fit using pandas idioms: center the data around its means so the
    intercept drops out, solve for weights, then recover the intercept."""

    def __init__(self, feature_columns: list[str], target_column: str = "y"):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.weights: pd.Series | None = None
        self.bias: float | None = None

    def fit(self, df: pd.DataFrame) -> "LinearRegressionOLS":
        log.debug(f"df shape={df.shape}")

        X, y = df[self.feature_columns], df[self.target_column]
        X_mean, y_mean = X.mean(), y.mean()
        X_centered, y_centered = (X - X_mean).values, (y - y_mean).values
        X_t = X_centered.transpose()
        X_sq = X_t @ X_centered

        # dRSS / d-beta = 0 => (X_t @ X) @ beta = X_t @ y
        weights = np.linalg.solve(X_sq, X_t @ y_centered)
        self.weights = pd.Series(weights, index=self.feature_columns)

        self.bias = y_mean - self.weights @ X_mean

        log.debug(f"weights={self.weights} bias={self.bias}")
        log.debug(f"shapes of weights={
                  self.weights.shape} bias={self.bias.shape}")

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")

        return df[self.feature_columns].dot(self.weights) + self.bias
