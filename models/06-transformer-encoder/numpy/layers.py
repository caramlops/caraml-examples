"""Generic, fully-implemented, hand-differentiated neural net layers --
Linear, LayerNorm, ReLU, token Embedding, the softmax+cross-entropy output
loss, sinusoidal positional encoding, and a plain Adam optimizer.

None of this is the paper's novel contribution (Vaswani et al. reuse all of
it, or treat it as a fixed, well-known building block), so none of it is a
placeholder -- see attention.py for the one piece that is.

Every layer follows the same tiny manual-autograd convention: forward(...)
returns (output, cache), and backward(dout, cache) returns the gradients
w.r.t. every input (including parameters) in the same order forward took
them. There's no computation graph tracking anything automatically --
that's exactly the "under the hood" thing you're meant to feel by hand
here, same spirit as torch/model_raw.py in 01-linear-regression, just for
a much bigger stack of operations.
"""

import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        scale = (2.0 / in_features) ** 0.5
        self.W = rng.normal(scale=scale, size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        y = x @ self.W + self.b
        return y, (x,)

    def backward(self, dy: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        (x,) = cache
        dx = dy @ self.W.T
        dW = x.T @ dy
        db = dy.sum(axis=0)
        return dx, dW, db


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu) * std_inv
        y = self.gamma * xhat + self.beta
        return y, (xhat, std_inv)

    def backward(self, dy: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xhat, std_inv = cache
        D = dy.shape[-1]
        dgamma = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
        dbeta = dy.sum(axis=tuple(range(dy.ndim - 1)))
        dxhat = dy * self.gamma
        dx = (
            std_inv
            / D
            * (
                D * dxhat
                - dxhat.sum(axis=-1, keepdims=True)
                - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
            )
        )
        return dx, dgamma, dbeta


class ReLU:
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        return np.maximum(0, x), (x,)

    def backward(self, dy: np.ndarray, cache: tuple) -> np.ndarray:
        (x,) = cache
        return dy * (x > 0)


class Embedding:
    def __init__(self, vocab_size: int, d_model: int, rng: np.random.Generator):
        self.W = rng.normal(scale=0.02, size=(vocab_size, d_model))

    def forward(self, ids: np.ndarray) -> tuple[np.ndarray, tuple]:
        return self.W[ids], (ids,)

    def backward(self, dy: np.ndarray, cache: tuple) -> np.ndarray:
        (ids,) = cache
        dW = np.zeros_like(self.W)
        np.add.at(dW, ids.reshape(-1), dy.reshape(-1, dy.shape[-1]))
        return dW


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(...)
    -- equations 3/4 of Vaswani et al. Fixed, not learned: no backward
    needed, since gradient just flows straight through the addition to
    the token embeddings beneath it."""
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """logits: (N, vocab_size), targets: (N,) integer class ids. Returns
    (mean loss, dlogits) -- forward and backward combined, same clean
    (probs - onehot) gradient as 03-logistic-regression's softmax +
    cross-entropy."""
    z = logits - logits.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=-1, keepdims=True)
    n = logits.shape[0]
    loss = -np.mean(np.log(probs[np.arange(n), targets] + 1e-12))
    dlogits = probs.copy()
    dlogits[np.arange(n), targets] -= 1
    dlogits /= n
    return float(loss), dlogits


class Adam:
    """Standard Adam (Kingma & Ba, 2014), operating on a flat dict of
    {name: array} parameters and matching gradients."""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
