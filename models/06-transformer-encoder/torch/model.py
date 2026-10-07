import logging
import math

import torch
from torch import nn

log = logging.getLogger(__name__)


class BidirectionalSelfAttention(nn.Module):
    """The placeholder in this runtime -- bidirectional self-attention
    with a padding mask, the encoder-side counterpart to
    05-transformer/torch/model.py's causal CausalSelfAttention. Same
    scaled dot-product mechanics; what's different is there's no causal
    mask at all (every position may attend to every other position), and
    instead a *padding* mask that varies per-sequence in the batch
    (sequences are padded to a fixed length, and [PAD] tokens must never
    be attended to)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D). padding_mask: (B, T) bool, True = real token."""
        log.debug(f"x shape={x.shape}")
        B, T, D = x.shape

        # TODO(you): implement scaled dot-product multi-head bidirectional
        # self-attention with a padding mask -- same steps 1-3/5-7 as
        # 05-transformer/torch/model.py's CausalSelfAttention, but step 4
        # (the mask) is different.
        #   1. q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #   2. Split into heads: .view(B, T, self.n_heads, self.d_head)
        #      then .transpose(1, 2) -> (B, n_heads, T, d_head).
        #   3. scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        #      -- shape (B, n_heads, T, T).
        #   4. Padding mask (the different part): reshape padding_mask
        #      from (B, T) to (B, 1, 1, T) so it broadcasts against
        #      `scores`' last axis -- masking is about which *keys* are
        #      real tokens, not about query position, and unlike 05's
        #      (T, T) causal mask, this one genuinely varies per sequence
        #      in the batch:
        #        mask = padding_mask[:, None, None, :]
        #        scores = scores.masked_fill(~mask, float("-inf"))
        #      No triangular restriction at all -- every valid key is
        #      attendable from every query position.
        #   5. attn = torch.softmax(scores, dim=-1)
        #   6. out = attn @ v
        #   7. Merge heads back: .transpose(1, 2).contiguous().view(B, T, D)
        #   8. return self.out_proj(out)
        raise NotImplementedError("Implement multi-head bidirectional self-attention")


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class BERTClassifier(nn.Module):
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
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", self._sinusoidal_positional_encoding(max_len, d_model))
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    @staticmethod
    def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        padding_mask = ids != self.pad_id
        x = self.token_emb(ids) + self.pos_encoding[:T]
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.ln_f(x)
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)
