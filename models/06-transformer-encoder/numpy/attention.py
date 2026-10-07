"""Multi-head *bidirectional* self-attention with a *padding* mask -- the
encoder-side counterpart to 05-transformer/numpy/attention.py's decoder
(causal) attention. Both the softmax-backward derivation and the overall
forward/backward structure are identical to that file (same scaled
dot-product mechanics, same clean softmax Jacobian); what's different, and
what you're actually practicing here, is the *masking*:

- 05's decoder attention masks based on *position*: query i may never
  attend to key j > i, the same restriction for every sequence in every
  batch (an (T, T) mask with no batch dimension).
- This encoder's attention masks based on *content*: query i may attend to
  ANY key j (no positional restriction at all -- "bidirectional" is the
  whole point, every token sees every other token, including ones after
  it), except that key j must not be a [PAD] token. Since different
  sequences in a batch are padded to different actual lengths, this mask
  genuinely varies per-sequence (shape (B, 1, 1, T), not just (T, T)).

See 05-transformer/PAPER.md and numpy/attention.py's comments for the full
derivation if you want the reasoning again; this file assumes you've
either done that one first or are comfortable re-deriving the same
softmax-backward trick here.
"""

import numpy as np


class BidirectionalSelfAttention:
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

    def forward(self, x: np.ndarray, padding_mask: np.ndarray) -> tuple[np.ndarray, tuple]:
        """x: (B, T, D). padding_mask: (B, T) boolean, True = real token,
        False = [PAD] -- True at position j means key j is allowed to be
        attended to. Returns (y, cache), y same shape as x."""
        # TODO(you): implement scaled dot-product multi-head bidirectional
        # self-attention with a padding mask.
        #
        # Let B, T, D = x.shape, H = self.n_heads, d_h = self.d_head.
        #
        #   1. Project and split into heads exactly like
        #      05-transformer/numpy/attention.py's forward steps 1-2:
        #      Q, K, V = x @ Wq, x @ Wk, x @ Wv, each reshaped+transposed
        #      to (B, H, T, d_h).
        #   2. scores = Q @ K.transpose(-1, -2) / sqrt(d_h), shape
        #      (B, H, T, T) -- identical to the decoder's scaling step.
        #   3. Padding mask (the part that's different from 05): reshape
        #      `padding_mask` from (B, T) to (B, 1, 1, T) so it
        #      broadcasts against `scores`' last axis (the *key* axis --
        #      masking is about which keys are real tokens, independent
        #      of which query position is asking). Set scores to -inf
        #      wherever that broadcast mask is False:
        #        scores = np.where(padding_mask[:, None, None, :], scores, -1e9)
        #      Note there's no upper-triangular restriction here at all --
        #      every valid (non-pad) key is attendable from every query
        #      position, which is the actual "bidirectional" in the name.
        #   4. A = softmax(scores, axis=-1) -- same numerically-stable
        #      softmax as everywhere else in this repo.
        #   5. O = A @ V, merge heads back to (B, T, D), then
        #      y = O @ self.Wo -- identical to 05's steps 6-8.
        #
        # Cache everything backward() will need: x, Q, K, V (post-split),
        # A, and the merged O (pre-Wo) -- same as 05's attention.py.
        raise NotImplementedError("Implement multi-head bidirectional self-attention forward")

    def backward(
        self, dy: np.ndarray, cache: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """dy: (B, T, D), gradient w.r.t. this layer's output. Returns
        (dx, dWq, dWk, dWv, dWo)."""
        # TODO(you): backprop through every step of forward(), in reverse
        # -- structurally identical to 05-transformer/numpy/attention.py's
        # backward(), since the padding mask needs no special handling
        # here either: masked (-inf) positions already have A ~ 0 from
        # the forward softmax, so their contribution to dscores comes out
        # ~0 automatically, the same way the causal mask's future
        # positions did in 05.
        #
        #   1. dWo = O.reshape(B*T, D).T @ dy.reshape(B*T, D)
        #      dO = dy @ Wo.T, un-merge to (B, H, T, d_h).
        #   2. dA = dO @ V.transpose(-1, -2)
        #      dV = A.transpose(-1, -2) @ dO
        #   3. dscores = A * (dA - sum(dA * A, axis=-1, keepdims=True))
        #   4. dQK = dscores / sqrt(d_h)
        #      dQ = dQK @ K
        #      dK = dQK.transpose(-1, -2) @ Q
        #   5. Merge dQ, dK, dV back to (B, T, D), then:
        #      dWq = x.reshape(B*T,D).T @ dQ.reshape(B*T,D), dx_q = dQ @ Wq.T
        #      dWk = x.reshape(B*T,D).T @ dK.reshape(B*T,D), dx_k = dK @ Wk.T
        #      dWv = x.reshape(B*T,D).T @ dV.reshape(B*T,D), dx_v = dV @ Wv.T
        #   6. dx = dx_q + dx_k + dx_v (x feeds all three projections).
        raise NotImplementedError("Implement multi-head bidirectional self-attention backward")
