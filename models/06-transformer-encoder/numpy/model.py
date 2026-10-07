"""Assembles the encoder-only transformer (BERT-style) from layers.py's
generic layers and attention.py's bidirectional attention. Fully
implemented -- the architecture's wiring isn't the paper's contribution,
bidirectional self-attention with a padding mask (attention.py) is.

Pre-LN residual structure, same choice (and same justification) as
05-transformer:

    x = token_embedding(ids) + sinusoidal_positional_encoding
    for each block:
        x = x + bidirectional_attention(layernorm(x), padding_mask)
        x = x + feedforward(layernorm(x))
    x = layernorm(x)
    cls_repr = x[:, 0, :]           # the [CLS] token's final representation
    logits = cls_repr @ W_head + b_head   # classification head

Vocab ids 0..vocab_size-1 are characters; vocab_size and vocab_size+1 are
reserved for [CLS] and [PAD] respectively (see data/make_dataset.py).
"""

import numpy as np

from attention import BidirectionalSelfAttention
from layers import Embedding, LayerNorm, Linear, ReLU, sinusoidal_positional_encoding


class TransformerEncoderBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng: np.random.Generator):
        self.ln1 = LayerNorm(d_model)
        self.attn = BidirectionalSelfAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.fc1 = Linear(d_model, d_ff, rng)
        self.relu = ReLU()
        self.fc2 = Linear(d_ff, d_model, rng)

    def forward(self, x: np.ndarray, padding_mask: np.ndarray) -> tuple[np.ndarray, tuple]:
        B, T, D = x.shape

        normed1, ln1_cache = self.ln1.forward(x)
        attn_out, attn_cache = self.attn.forward(normed1, padding_mask)
        x1 = x + attn_out

        normed2, ln2_cache = self.ln2.forward(x1)
        h, fc1_cache = self.fc1.forward(normed2.reshape(B * T, D))
        h_relu, relu_cache = self.relu.forward(h)
        ff_out, fc2_cache = self.fc2.forward(h_relu)
        x2 = x1 + ff_out.reshape(B, T, D)

        cache = (ln1_cache, attn_cache, ln2_cache, fc1_cache, relu_cache, fc2_cache, B, T, D)
        return x2, cache

    def backward(self, dx2: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict]:
        ln1_cache, attn_cache, ln2_cache, fc1_cache, relu_cache, fc2_cache, B, T, D = cache

        dff_out = dx2.reshape(B * T, D)
        dh_relu, dW_fc2, db_fc2 = self.fc2.backward(dff_out, fc2_cache)
        dh = self.relu.backward(dh_relu, relu_cache)
        dnormed2, dW_fc1, db_fc1 = self.fc1.backward(dh, fc1_cache)
        dx1_from_ff, dgamma2, dbeta2 = self.ln2.backward(dnormed2.reshape(B, T, D), ln2_cache)
        dx1 = dx2 + dx1_from_ff

        dnormed1, dWq, dWk, dWv, dWo = self.attn.backward(dx1, attn_cache)
        dx_from_attn, dgamma1, dbeta1 = self.ln1.backward(dnormed1, ln1_cache)
        dx = dx1 + dx_from_attn

        grads = {
            "ln1.gamma": dgamma1,
            "ln1.beta": dbeta1,
            "attn.Wq": dWq,
            "attn.Wk": dWk,
            "attn.Wv": dWv,
            "attn.Wo": dWo,
            "ln2.gamma": dgamma2,
            "ln2.beta": dbeta2,
            "fc1.W": dW_fc1,
            "fc1.b": db_fc1,
            "fc2.W": dW_fc2,
            "fc2.b": db_fc2,
        }
        return dx, grads


class BERTClassifier:
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
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size  # includes [CLS]/[PAD] (see make_dataset.py's embedding table sizing)
        self.max_len = max_len
        self.pad_id = pad_id
        self.token_emb = Embedding(vocab_size, d_model, rng)
        self.pos_encoding = sinusoidal_positional_encoding(max_len, d_model)
        self.blocks = [TransformerEncoderBlock(d_model, n_heads, d_ff, rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.classifier = Linear(d_model, n_classes, rng)

    def forward(self, ids: np.ndarray) -> tuple[np.ndarray, tuple]:
        """ids: (B, T) integer token ids (T = max_len, [CLS] at position
        0). Returns (logits, cache), logits shape (B, n_classes)."""
        B, T = ids.shape
        padding_mask = ids != self.pad_id  # (B, T), True = real token

        tok_emb, tok_cache = self.token_emb.forward(ids)
        x = tok_emb + self.pos_encoding[:T][None, :, :]

        block_caches = []
        for block in self.blocks:
            x, c = block.forward(x, padding_mask)
            block_caches.append(c)

        x_normed, lnf_cache = self.ln_f.forward(x)
        cls_repr = x_normed[:, 0, :]  # (B, D) -- the [CLS] token's final representation
        logits, head_cache = self.classifier.forward(cls_repr)

        cache = (tok_cache, block_caches, lnf_cache, head_cache, B, T)
        return logits, cache

    def backward(self, dlogits: np.ndarray, cache: tuple) -> dict:
        """dlogits: (B, n_classes). Returns a flat {param_name: grad}
        dict matching self.params()'s keys."""
        tok_cache, block_caches, lnf_cache, head_cache, B, T = cache

        dcls_repr, dW_head, db_head = self.classifier.backward(dlogits, head_cache)

        D = dcls_repr.shape[-1]
        dx_normed = np.zeros((B, T, D))
        dx_normed[:, 0, :] = dcls_repr  # gradient only flows back through the [CLS] position

        dx, dgamma_f, dbeta_f = self.ln_f.backward(dx_normed, lnf_cache)

        grads = {
            "classifier.W": dW_head,
            "classifier.b": db_head,
            "ln_f.gamma": dgamma_f,
            "ln_f.beta": dbeta_f,
        }
        for i in reversed(range(len(self.blocks))):
            dx, block_grads = self.blocks[i].backward(dx, block_caches[i])
            for name, g in block_grads.items():
                grads[f"block{i}.{name}"] = g

        grads["token_emb.W"] = self.token_emb.backward(dx, tok_cache)
        return grads

    def params(self) -> dict[str, np.ndarray]:
        params = {
            "classifier.W": self.classifier.W,
            "classifier.b": self.classifier.b,
            "ln_f.gamma": self.ln_f.gamma,
            "ln_f.beta": self.ln_f.beta,
            "token_emb.W": self.token_emb.W,
        }
        for i, block in enumerate(self.blocks):
            params[f"block{i}.ln1.gamma"] = block.ln1.gamma
            params[f"block{i}.ln1.beta"] = block.ln1.beta
            params[f"block{i}.attn.Wq"] = block.attn.Wq
            params[f"block{i}.attn.Wk"] = block.attn.Wk
            params[f"block{i}.attn.Wv"] = block.attn.Wv
            params[f"block{i}.attn.Wo"] = block.attn.Wo
            params[f"block{i}.ln2.gamma"] = block.ln2.gamma
            params[f"block{i}.ln2.beta"] = block.ln2.beta
            params[f"block{i}.fc1.W"] = block.fc1.W
            params[f"block{i}.fc1.b"] = block.fc1.b
            params[f"block{i}.fc2.W"] = block.fc2.W
            params[f"block{i}.fc2.b"] = block.fc2.b
        return params
