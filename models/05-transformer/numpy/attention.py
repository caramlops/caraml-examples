"""Multi-head causal self-attention -- the one piece of this model that's a
placeholder, because it's the actual contribution of Vaswani et al.,
"Attention Is All You Need" (2017). Every other layer in this model
(layers.py) is a well-known, pre-existing building block the paper reuses;
this is the mechanism the paper introduces.

Implement both forward() and backward() by hand -- there's no autograd in
this runtime, so the backward pass is a derivation you do yourself, not
something a framework gives you for free. Once you've implemented both,
run `uv run python models/05-transformer/numpy/test_gradients.py` to
numerically verify your backward pass against finite differences on a
tiny random instance, before trusting it inside a real training run.
"""

import numpy as np


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        scale = (2.0 / d_model) ** 0.5
        self.Wq = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wk = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wv = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wo = rng.normal(scale=scale, size=(d_model, d_model))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        """x: (batch, seq_len, d_model). Returns (y, cache), y same shape
        as x."""
        # TODO(you): implement scaled dot-product multi-head causal
        # self-attention -- Vaswani et al. eq. 1 (per head) and eq. 2
        # (concatenating heads), with a causal mask so position i can only
        # attend to positions <= i.
        #
        # Let B, T, D = x.shape, H = self.n_heads, d_h = self.d_head.
        #
        #   1. Project: Q = x @ self.Wq, K = x @ self.Wk, V = x @ self.Wv
        #      -- each shape (B, T, D).
        #   2. Split into heads: reshape each to (B, T, H, d_h), then
        #      transpose to (B, H, T, d_h) so the matmuls below operate
        #      per-head, batched over (B, H).
        #   3. Scaled dot-product scores:
        #      scores = Q @ K.transpose(-1, -2) / sqrt(d_h)
        #      shape (B, H, T, T) -- scores[..., i, j] is how much
        #      position i attends to position j, before masking/softmax.
        #   4. Causal mask: position i must not attend to position j > i.
        #      Build a (T, T) mask (e.g. via np.triu with k=1) and set
        #      those entries of `scores` to -inf (or a very large
        #      negative number) before the softmax, so they get exactly
        #      zero attention weight.
        #   5. Softmax over the last axis: A = softmax(scores, axis=-1)
        #      -- same numerically-stable softmax as
        #      03-logistic-regression (subtract the row max first). A is
        #      the actual attention-weight matrix, shape (B, H, T, T).
        #   6. Weighted sum of values: O = A @ V, shape (B, H, T, d_h).
        #   7. Merge heads back: transpose O to (B, T, H, d_h), reshape to
        #      (B, T, D).
        #   8. Output projection: y = O @ self.Wo.
        #
        # Cache everything backward() will need: at minimum x, Q, K, V
        # (post-split, shape (B, H, T, d_h)), A, and the merged O
        # (pre-Wo, shape (B, T, D)).
        raise NotImplementedError("Implement multi-head causal self-attention forward")

    def backward(
        self, dy: np.ndarray, cache: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """dy: (batch, seq_len, d_model), gradient w.r.t. this layer's
        output. Returns (dx, dWq, dWk, dWv, dWo)."""
        # TODO(you): backprop through every step of forward(), in reverse.
        #
        #   1. Through the output projection (y = O @ Wo):
        #      dWo = O.reshape(B * T, D).T @ dy.reshape(B * T, D)
        #      dO  = dy @ Wo.T, then un-merge back to (B, H, T, d_h) the
        #      same way you merged it in forward (reshape + transpose,
        #      done in reverse).
        #   2. Through the weighted sum (O = A @ V):
        #      dA = dO @ V.transpose(-1, -2)
        #      dV = A.transpose(-1, -2) @ dO
        #   3. Through the softmax (A = softmax(scores)): for each row,
        #      the softmax Jacobian gives
        #        dscores = A * (dA - sum(dA * A, axis=-1, keepdims=True))
        #      This is the same "diag(a) - a a^T" softmax-gradient
        #      structure as 03-logistic-regression's (probs - onehot) --
        #      just applied per attention row here instead of per
        #      classification row, and derived from dA directly rather
        #      than from a loss. The masked (-inf) positions take care of
        #      themselves: their A is already ~0, so their contribution to
        #      dscores comes out ~0 too, with no extra masking needed here.
        #   4. Through the scale (scores = QK^T / sqrt(d_h)):
        #      dQK = dscores / sqrt(d_h), where QK = Q @ K.transpose(-1,-2)
        #      dQ = dQK @ K
        #      dK = dQK.transpose(-1, -2) @ Q
        #   5. Merge dQ, dK, dV back from (B, H, T, d_h) to (B, T, D) the
        #      same way as step 7 of forward, then:
        #      dWq = x.reshape(B*T, D).T @ dQ.reshape(B*T, D), dx_q = dQ @ Wq.T
        #      dWk = x.reshape(B*T, D).T @ dK.reshape(B*T, D), dx_k = dK @ Wk.T
        #      dWv = x.reshape(B*T, D).T @ dV.reshape(B*T, D), dx_v = dV @ Wv.T
        #   6. x feeds all three projections, so its total gradient is the
        #      sum of all three branches: dx = dx_q + dx_k + dx_v.
        raise NotImplementedError("Implement multi-head causal self-attention backward")
