import logging

import pandas as pd

log = logging.getLogger(__name__)


class RidgeRegression:
    """Ridge fit using pandas idioms: center the data around its means (as
    in OLS) so the intercept drops out and stays unregularized, then solve
    the penalized normal equation on the centered data."""

    def __init__(self, feature_columns: list[str], alpha: float = 1.0, target_column: str = "y"):
        self.feature_columns = feature_columns
        self.alpha = alpha
        self.target_column = target_column
        self.weights: pd.Series | None = None
        self.bias: float | None = None

    def fit(self, df: pd.DataFrame) -> "RidgeRegression":
        log.debug(f"df shape={df.shape}")

        # TODO(you): implement ridge via mean-centering.
        #
        #   1. X = df[self.feature_columns], y = df[self.target_column].
        #   2. Center both around their column means (X.mean(), y.mean()).
        #      Note: centering already removes the intercept from the
        #      problem, so unlike the numpy runtime there's no diagonal
        #      entry to zero out -- every remaining weight is a feature
        #      weight, all fair game for shrinkage.
        #   3. Solve the ridge normal equation on the centered data:
        #         weights = (Xc^T Xc + self.alpha * I)^-1 Xc^T yc
        #      (pull `.values` and use numpy for the linear algebra, as in
        #      the plain-OLS pandas runtime).
        #   4. Recover the (unregularized) intercept:
        #         bias = y.mean() - weights @ X.mean()
        #   5. Store weights as a pd.Series indexed by self.feature_columns.
        raise NotImplementedError("Implement the ridge solver using pandas + numpy")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return df[self.feature_columns].dot(self.weights) + self.bias
