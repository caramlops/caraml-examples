import logging

import numpy as np

log = logging.getLogger(__name__)


class PoissonRegression:
    """Poisson regression (count data, log link), fit via IRLS --
    iteratively reweighted least squares. Unlike 03-logistic-regression's
    plain gradient descent, IRLS is Newton's-method-flavored: each
    iteration solves a *weighted* version of the exact normal equation
    from 01-linear-regression, reweighting by how much the current fit
    trusts each sample, and converges in a handful of iterations instead
    of thousands."""

    def __init__(self, n_features: int, n_iters: int = 25):
        self.n_iters = n_iters
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def _working_response(self, eta: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log.debug(f"eta shape={eta.shape}")

        # TODO(you): compute the GLM "working" quantities for one IRLS
        # iteration, for a Poisson outcome with a log link.
        #   mu = exp(eta)   -- the inverse link: predicted mean count.
        #   W  = mu         -- the IRLS weight. In general, the weight for
        #        one IRLS step is 1 / (Var(mu) * g'(mu)^2), where V is the
        #        variance function and g is the link. For Poisson,
        #        Var(mu) = mu; for the log link, g(mu) = log(mu), so
        #        g'(mu) = 1/mu. Substituting: 1 / (mu * (1/mu)^2) = mu --
        #        the weight collapses to exactly mu. (This clean
        #        simplification is specific to the log link being
        #        Poisson's *canonical* link -- the same reason
        #        softmax + cross-entropy's combined gradient was so clean
        #        in 03-logistic-regression.)
        #   z  = eta + (y - mu) / mu   -- the "working response": a
        #        linearized pseudo-target that ordinary weighted-least-
        #        squares machinery can be pointed at.
        # Return (mu, W, z), all shape (n_samples,).
        raise NotImplementedError("Implement the Poisson/log-link working quantities")

    def _irls_step(self, X_aug: np.ndarray, W: np.ndarray, z: np.ndarray) -> np.ndarray:
        log.debug(f"X_aug shape={X_aug.shape}")

        # TODO(you): solve one weighted normal equation -- the IRLS
        # update. This is the exact same equation as
        # 01-linear-regression's OLS solve, with X_aug and z scaled by W
        # first:
        #   (X_aug^T @ diag(W) @ X_aug) @ w_aug = X_aug^T @ diag(W) @ z
        # Don't build the full (n_samples, n_samples) diagonal matrix --
        # `X_aug * W[:, None]` scales each row of X_aug by its own weight
        # directly, which is exactly what `X_aug^T @ diag(W)` means,
        # without the O(n^2) memory of an explicit diagonal matrix.
        #   A = X_aug.T @ (X_aug * W[:, None])
        #   b = X_aug.T @ (W * z)
        #   return np.linalg.solve(A, b)
        raise NotImplementedError("Implement the weighted normal-equation IRLS step")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PoissonRegression":
        n_samples, n_features = X.shape
        X_aug = np.c_[X, np.ones(n_samples)]

        # Start the intercept at log(mean count) rather than 0 -- a
        # reasonable guess for the "no predictors" case -- so the first
        # iteration's working response doesn't have to recover from a
        # wild initial mu=1 for every sample.
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
