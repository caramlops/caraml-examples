import logging

import numpy as np

log = logging.getLogger(__name__)


class LogisticRegression:
    """Multinomial (softmax) logistic regression, fit via batch gradient
    descent on the cross-entropy loss. Unlike OLS, there is no closed-form
    solution here -- the maximum-likelihood equations are transcendental,
    not linear -- so gradient descent isn't "an alternative," it's the
    only option, for every runtime."""

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.5, n_iters: int = 2000):
        self.n_classes = n_classes
        self.lr = lr
        self.n_iters = n_iters
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        log.debug(f"logits shape={logits.shape}")

        # TODO(you): implement a *numerically stable* softmax over the
        # last axis.
        #   - logits has shape (n_samples, n_classes).
        #   - softmax(z)_k = exp(z_k) / sum_j exp(z_j), applied per row.
        #   - Subtract each row's max from itself before exponentiating:
        #     softmax(z) == softmax(z - max(z)) mathematically (the max
        #     cancels out in the ratio), but without it, exp() of a
        #     moderately large logit overflows to inf and every
        #     probability comes out nan. This is the single most common
        #     bug in a from-scratch softmax.
        #   - Return shape (n_samples, n_classes), each row summing to 1.
        raise NotImplementedError("Implement a numerically stable softmax")

    def _step(self, X: np.ndarray, probs: np.ndarray, Y_onehot: np.ndarray) -> None:
        log.debug(f"weights shape={self.weights.shape}")

        # TODO(you): compute the gradient of the mean cross-entropy loss
        # w.r.t. weights and bias, then take one gradient-descent step.
        #   cross-entropy loss = -mean(sum(Y_onehot * log(probs), axis=1))
        #   The gradient of that loss w.r.t. the *logits* has a famously
        #   clean closed form: (probs - Y_onehot) / n_samples -- softmax
        #   and cross-entropy are almost always paired specifically
        #   because their combined derivative simplifies this much.
        #   grad_weights = X.T @ (probs - Y_onehot) / n_samples   # (n_features, n_classes)
        #   grad_bias    = (probs - Y_onehot).mean(axis=0)         # (n_classes,)
        #   self.weights -= self.lr * grad_weights
        #   self.bias    -= self.lr * grad_bias
        raise NotImplementedError("Implement one gradient-descent step")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        Y_onehot = np.eye(self.n_classes)[y]
        for i in range(self.n_iters):
            logits = X @ self.weights + self.bias
            probs = self._softmax(logits)
            self._step(X, probs, Y_onehot)
            if i % 200 == 0:
                loss = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-12), axis=1))
                log.debug(f"iter={i} loss={loss:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
