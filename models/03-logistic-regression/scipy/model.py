import logging

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)


class LogisticRegression:
    """Multinomial logistic regression fit via scipy.optimize.minimize.
    Unlike OLS/ridge, there's no normal equation here at all -- the
    maximum-likelihood equations are transcendental, not linear -- so a
    general-purpose optimizer is the *natural* tool for this model, not
    just an alternative to a closed form that doesn't exist."""

    def __init__(self, n_features: int, n_classes: int):
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_w = self.n_features * self.n_classes
        weights = params[:n_w].reshape(self.n_features, self.n_classes)
        bias = params[n_w:]
        return weights, bias

    def _loss_and_grad(
        self, params: np.ndarray, X: np.ndarray, Y_onehot: np.ndarray
    ) -> tuple[float, np.ndarray]:
        log.debug(f"params shape={params.shape}")
        weights, bias = self._unpack(params)

        # TODO(you): compute both the mean cross-entropy loss and its
        # gradient w.r.t. `params`, and return them as (loss, grad) --
        # scipy.optimize.minimize(..., jac=True) expects exactly that pair
        # from a single function call, rather than two separate functions.
        #   n_samples = X.shape[0]
        #   logits = X @ weights + bias                      # (n_samples, n_classes)
        #   probs  = softmax(logits) -- subtract each row's max before
        #            exponentiating, same numerical-stability reason as
        #            every other runtime's softmax.
        #   loss = -mean(sum(Y_onehot * log(probs), axis=1))
        #   grad_weights = X.T @ (probs - Y_onehot) / n_samples   # (n_features, n_classes)
        #   grad_bias    = (probs - Y_onehot).mean(axis=0)         # (n_classes,)
        #   grad = np.concatenate([grad_weights.ravel(), grad_bias])  # flatten to match `params`
        raise NotImplementedError("Implement the cross-entropy loss and gradient")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        log.debug(f"X shape={X.shape}, y shape={y.shape}")
        Y_onehot = np.eye(self.n_classes)[y]
        x0 = np.zeros(self.n_features * self.n_classes + self.n_classes)
        result = minimize(self._loss_and_grad, x0, args=(X, Y_onehot), jac=True, method="L-BFGS-B")
        self.weights, self.bias = self._unpack(result.x)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights + self.bias
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
