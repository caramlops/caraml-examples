import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class LogisticRegression:
    """Same softmax regression as the numpy runtime. Unlike OLS, logistic
    regression has no closed form at all, so there's no alternative
    derivation to practice here the way pandas' mean-centering trick was
    for linear regression -- this is the same gradient-descent algorithm,
    expressed with pandas idioms on the data-handling side: reading
    columns by name, returning predictions as a labeled DataFrame/Series
    instead of a bare array."""

    def __init__(
        self,
        feature_columns: list[str],
        target_column: str = "y",
        n_classes: int = 3,
        lr: float = 0.5,
        n_iters: int = 2000,
    ):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.n_classes = n_classes
        self.lr = lr
        self.n_iters = n_iters
        self.weights: pd.DataFrame | None = None
        self.bias: pd.Series | None = None
        self._w: np.ndarray | None = None
        self._b: np.ndarray | None = None

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        log.debug(f"logits shape={logits.shape}")

        # TODO(you): same as the numpy runtime -- implement a numerically
        # stable softmax over the last axis (subtract each row's max
        # before exponentiating). Operate on the raw numpy array; pandas
        # has no softmax of its own.
        raise NotImplementedError("Implement a numerically stable softmax")

    def _step(self, X: np.ndarray, probs: np.ndarray, Y_onehot: np.ndarray) -> None:
        log.debug(f"weights shape={self._w.shape}")

        # TODO(you): same cross-entropy gradient as the numpy runtime,
        # mutating self._w / self._b in place.
        #   grad_weights = X.T @ (probs - Y_onehot) / n_samples
        #   grad_bias    = (probs - Y_onehot).mean(axis=0)
        #   self._w -= self.lr * grad_weights
        #   self._b -= self.lr * grad_bias
        raise NotImplementedError("Implement one gradient-descent step")

    def fit(self, df: pd.DataFrame) -> "LogisticRegression":
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        n_features = X.shape[1]
        Y_onehot = np.eye(self.n_classes)[y]

        self._w = np.zeros((n_features, self.n_classes))
        self._b = np.zeros(self.n_classes)

        for i in range(self.n_iters):
            logits = X @ self._w + self._b
            probs = self._softmax(logits)
            self._step(X, probs, Y_onehot)
            if i % 200 == 0:
                loss = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-12), axis=1))
                log.debug(f"iter={i} loss={loss:.4f}")

        self.weights = pd.DataFrame(self._w, index=self.feature_columns)
        self.bias = pd.Series(self._b)
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict_proba()")
        logits = df[self.feature_columns].values @ self.weights.values + self.bias.values
        probs = self._softmax(logits)
        return pd.DataFrame(probs, index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return self.predict_proba(df).idxmax(axis=1)
