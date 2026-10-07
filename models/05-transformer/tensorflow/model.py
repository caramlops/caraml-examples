import logging
import math

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


class CausalSelfAttention(tf.Module):
    """The one placeholder in this runtime -- multi-head causal
    self-attention, same as every other runtime here. GradientTape
    handles the backward pass for you; you only need the forward
    computation, written out explicitly rather than reached for as a
    single fused builtin."""

    def __init__(self, d_model: int, n_heads: int, block_size: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        init = tf.keras.initializers.GlorotUniform()
        self.Wq = tf.Variable(init([d_model, d_model]), name="Wq")
        self.Wk = tf.Variable(init([d_model, d_model]), name="Wk")
        self.Wv = tf.Variable(init([d_model, d_model]), name="Wv")
        self.Wo = tf.Variable(init([d_model, d_model]), name="Wo")
        self.causal_mask = tf.constant(np.tril(np.ones((block_size, block_size), dtype=bool)))

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        log.debug(f"x shape={x.shape}")
        B, T = tf.shape(x)[0], tf.shape(x)[1]

        # TODO(you): implement scaled dot-product multi-head causal
        # self-attention -- same math as every other runtime here.
        #   1. Project: q = x @ self.Wq, k = x @ self.Wk, v = x @ self.Wv
        #      -- each shape (B, T, D).
        #   2. Split into heads: tf.reshape to (B, T, n_heads, d_head),
        #      then tf.transpose with perm=[0, 2, 1, 3] to get shape
        #      (B, n_heads, T, d_head).
        #   3. scores = q @ tf.transpose(k, perm=[0, 1, 3, 2]) /
        #      tf.sqrt(tf.cast(self.d_head, tf.float32)) -- shape
        #      (B, n_heads, T, T).
        #   4. Causal mask: self.causal_mask[:T, :T] is True where
        #      position i may attend to j (j <= i). Use
        #      tf.where(mask, scores, -1e9) (broadcasting the (T, T) mask
        #      against the (B, n_heads, T, T) scores) to suppress
        #      attention to future positions before the softmax.
        #   5. attn = tf.nn.softmax(scores, axis=-1)
        #   6. out = attn @ v -- shape (B, n_heads, T, d_head).
        #   7. Merge heads back: tf.transpose(out, perm=[0, 2, 1, 3]),
        #      then tf.reshape to (B, T, D).
        #   8. return out @ self.Wo
        raise NotImplementedError("Implement multi-head causal self-attention")


class TransformerBlock(tf.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, block_size: int):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.fc2 = tf.keras.layers.Dense(d_model)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(self.fc1(self.ln2(x)))
        return x


class GPT(tf.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_emb = tf.Variable(
            tf.keras.initializers.GlorotUniform()([vocab_size, d_model]), name="token_emb"
        )
        self.pos_encoding = tf.constant(
            self._sinusoidal_positional_encoding(block_size, d_model), dtype=tf.float32
        )
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, block_size) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization()
        self.lm_head = tf.keras.layers.Dense(vocab_size)

    @staticmethod
    def _sinusoidal_positional_encoding(block_size: int, d_model: int) -> np.ndarray:
        position = np.arange(block_size)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = np.zeros((block_size, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def __call__(self, ids: tf.Tensor) -> tf.Tensor:
        T = tf.shape(ids)[1]
        x = tf.gather(self.token_emb, ids) + self.pos_encoding[:T]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
