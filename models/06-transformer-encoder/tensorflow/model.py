import logging
import math

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


class BidirectionalSelfAttention(tf.Module):
    """The placeholder in this runtime -- bidirectional self-attention
    with a padding mask. Same idea as 05-transformer's causal attention,
    with the masking logic swapped: no triangular restriction, but a
    per-sequence padding mask on the key axis instead."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        init = tf.keras.initializers.GlorotUniform()
        self.Wq = tf.Variable(init([d_model, d_model]), name="Wq")
        self.Wk = tf.Variable(init([d_model, d_model]), name="Wk")
        self.Wv = tf.Variable(init([d_model, d_model]), name="Wv")
        self.Wo = tf.Variable(init([d_model, d_model]), name="Wo")

    def __call__(self, x: tf.Tensor, padding_mask: tf.Tensor) -> tf.Tensor:
        """x: (B, T, D). padding_mask: (B, T) bool, True = real token."""
        log.debug(f"x shape={x.shape}")
        B, T = tf.shape(x)[0], tf.shape(x)[1]

        # TODO(you): implement scaled dot-product multi-head bidirectional
        # self-attention with a padding mask -- same steps as
        # 05-transformer/tensorflow/model.py's attention, but step 4 (the
        # mask) is different.
        #   1. q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        #   2. Split into heads: tf.reshape to (B, T, n_heads, d_head),
        #      tf.transpose with perm=[0, 2, 1, 3] -> (B, n_heads, T, d_head)
        #   3. scores = q @ tf.transpose(k, perm=[0, 1, 3, 2]) /
        #      tf.sqrt(tf.cast(self.d_head, tf.float32))
        #   4. Padding mask (the different part): reshape padding_mask
        #      from (B, T) to (B, 1, 1, T) to broadcast against `scores`'
        #      last axis -- no triangular restriction at all, every valid
        #      key is attendable from every query position:
        #        mask = padding_mask[:, None, None, :]
        #        scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))
        #   5. attn = tf.nn.softmax(scores, axis=-1)
        #   6. out = attn @ v
        #   7. Merge heads: tf.transpose(out, perm=[0, 2, 1, 3]), then
        #      tf.reshape to (B, T, D)
        #   8. return out @ self.Wo
        raise NotImplementedError("Implement multi-head bidirectional self-attention")


class TransformerEncoderBlock(tf.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.attn = BidirectionalSelfAttention(d_model, n_heads)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.fc2 = tf.keras.layers.Dense(d_model)

    def __call__(self, x: tf.Tensor, padding_mask: tf.Tensor) -> tf.Tensor:
        x = x + self.attn(self.ln1(x), padding_mask)
        x = x + self.fc2(self.fc1(self.ln2(x)))
        return x


class BERTClassifier(tf.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        n_classes: int,
        pad_id: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
    ):
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id
        self.token_emb = tf.Variable(
            tf.keras.initializers.GlorotUniform()([vocab_size, d_model]), name="token_emb"
        )
        self.pos_encoding = tf.constant(
            self._sinusoidal_positional_encoding(max_len, d_model), dtype=tf.float32
        )
        self.blocks = [TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization()
        self.classifier = tf.keras.layers.Dense(n_classes)

    @staticmethod
    def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
        position = np.arange(max_len)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def __call__(self, ids: tf.Tensor) -> tf.Tensor:
        T = tf.shape(ids)[1]
        padding_mask = ids != self.pad_id
        x = tf.gather(self.token_emb, ids) + self.pos_encoding[:T]
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.ln_f(x)
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)
