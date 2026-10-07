import logging

import numpy as np
from scipy import linalg

log = logging.getLogger(__name__)


class PoissonRegression:
    """Same IRLS fit as the numpy runtime, but the weighted normal
    equation at each iteration is solved via scipy.linalg.solve instead
    of numpy.linalg.solve -- the same scipy.linalg idiom as
    01-linear-regression's scipy runtime, just inside an iterative loop
    this time instead of a single solve."""

    def __init__(self, n_features: int, n_iters: int = 25):
        self.n_iters = n_iters
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def _working_response(self, eta: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log.debug(f"eta shape={eta.shape}")

        # TODO(you): same as the numpy runtime -- the Poisson/log-link
        # working quantities for one IRLS iteration.
        #   mu = exp(eta)
        #   W  = mu
        #   z  = eta + (y - mu) / mu
        # Return (mu, W, z).
        raise NotImplementedError("Implement the Poisson/log-link working quantities")

    def _irls_step(self, X_aug: np.ndarray, W: np.ndarray, z: np.ndarray) -> np.ndarray:
        log.debug(f"X_aug shape={X_aug.shape}")

        # TODO(you): solve one weighted normal equation using
        # scipy.linalg.solve instead of numpy.linalg.solve.
        #   A = X_aug.T @ (X_aug * W[:, None])   -- symmetric positive-
        #       definite, same reasoning as 01-linear-regression's scipy
        #       runtime, so assume_a="pos" applies here too.
        #   b = X_aug.T @ (W * z)
        #   return linalg.solve(A, b, assume_a="pos")
        raise NotImplementedError("Implement the weighted normal-equation IRLS step")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PoissonRegression":
        n_samples, n_features = X.shape
        X_aug = np.c_[X, np.ones(n_samples)]

        w_aug = np.zeros(n_features + 1)
        w_aug[-1] = np.log(y.mean() + 1e-8)

        for i in range(self.n_iters):
            eta = X_aug @ w_aug
            mu, W, z = self._working_response(eta, y)
            w_aug = self._irls_step(X_aug, W, z)
            if i % 5 == 0:
                deviance = 2 * np.sum(y * np.log((y + 1e-12) / mu) - (y - mu))
                log.debug(f"iter={i} deviance={deviance:.4f}")

        self.weights = w_aug[:-1]
        self.bias = float(w_aug[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit() before predict()")
        return np.exp(X @ self.weights + self.bias)
