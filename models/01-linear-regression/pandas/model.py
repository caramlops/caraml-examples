import pandas as pd


class LinearRegressionOLS:
    """OLS fit using pandas idioms: center the data around its means so the
    intercept drops out, solve for weights, then recover the intercept."""

    def __init__(self, feature_columns: list[str], target_column: str = "y"):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.weights: pd.Series | None = None
        self.bias: float | None = None

    def fit(self, df: pd.DataFrame) -> "LinearRegressionOLS":
        # TODO(you): implement OLS via mean-centering.
        #
        #   1. X = df[self.feature_columns], y = df[self.target_column].
        #   2. Center both around their column means (X.mean(), y.mean()) —
        #      this removes the need for a separate intercept column.
        #   3. Solve the normal equation on the centered data for the weight
        #      vector. Pandas has no linear algebra of its own, so pull out
        #      `.values` and use numpy (`np.linalg.solve` or `.lstsq`) here.
        #   4. Recover the intercept: bias = y.mean() - weights @ X.mean().
        #   5. Store weights as a pd.Series indexed by self.feature_columns.
        raise NotImplementedError("Implement the OLS solver using pandas + numpy")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return df[self.feature_columns].dot(self.weights) + self.bias
