import logging
import math

import torch
from torch import nn

log = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """The one placeholder in this runtime -- multi-head causal
    self-attention, Vaswani et al.'s actual contribution. Autograd handles
    the backward pass for you here (that's the whole point of using
    torch instead of numpy for this runtime); you only need the forward
    computation, written out explicitly rather than reached for as a
    single fused builtin, so the mechanism itself is what you practice."""

    def __init__(self, d_model: int, n_heads: int, block_size: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        causal_mask = torch.tril(torch.ones(block_size, block_size)).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log.debug(f"x shape={x.shape}")
        B, T, D = x.shape

        # TODO(you): implement scaled dot-product multi-head causal
        # self-attention -- Vaswani et al. eq. 1/2.
        #   1. Project: q = self.q_proj(x), k = self.k_proj(x),
        #      v = self.v_proj(x) -- each shape (B, T, D).
        #   2. Split into heads: .view(B, T, self.n_heads, self.d_head)
        #      then .transpose(1, 2) to get shape (B, n_heads, T, d_head),
        #      so the matmuls below operate per-head, batched over
        #      (B, n_heads).
        #   3. scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        #      -- shape (B, n_heads, T, T).
        #   4. Causal mask: self.causal_mask[:T, :T] is True where
        #      position i may attend to position j (j <= i). Use
        #      scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        #      to zero out (via softmax) attention to future positions.
        #   5. attn = torch.softmax(scores, dim=-1)
        #   6. out = attn @ v -- shape (B, n_heads, T, d_head).
        #   7. Merge heads back: .transpose(1, 2).contiguous().view(B, T, D)
        #   8. return self.out_proj(out)
        raise NotImplementedError("Implement multi-head causal self-attention")


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
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
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", self._sinusoidal_positional_encoding(block_size, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, block_size) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _sinusoidal_positional_encoding(block_size: int, d_model: int) -> torch.Tensor:
        position = torch.arange(block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(block_size, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.token_emb(ids) + self.pos_encoding[:T]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
