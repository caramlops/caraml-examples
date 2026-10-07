# 06 · Transformer encoder (BERT-style classifier)

A small text classifier: given a message, predict `ham` or `spam` via
bidirectional self-attention over its characters, pooled through a `[CLS]`
token. See `PAPER.md` for the paper this reproduces (Devlin et al.'s BERT,
building on `05-transformer`'s shared attention mechanism), the exact
mechanisms, and what was deliberately scoped down.

This is `05-transformer`'s sibling, not a repeat of it: same underlying
scaled dot-product attention math, but **bidirectional** (every position
attends to every other position, no causal restriction) with a
**padding** mask instead of a causal one (real messages vary in length;
sequences are padded to a fixed length and the pad tokens must never be
attended to). The data is the real SMS Spam Collection dataset
(`data/make_dataset.py` downloads it once) — a classifier trained on
synthetic text-label pairs wouldn't test whether the architecture can
learn a genuine classification signal.

| Runtime | File(s) | Placeholder(s) | Fully implemented |
|---|---|---|---|
| numpy | `numpy/attention.py` | bidirectional self-attention with a padding mask, **forward and backward both** | `numpy/layers.py`, `numpy/model.py` (architecture + `[CLS]` pooling), training loop, `numpy/test_gradients.py` |
| torch | `torch/model.py`, `torch/train.py` | attention forward pass (padding-masked, no causal mask); the training step | architecture wiring, data load, save/load |
| tensorflow | `tensorflow/model.py`, `tensorflow/train.py` | attention forward pass; the training step | architecture wiring, data load, save/load |
| onnx | `onnx/export.py`, `onnx/infer.py` | `torch.onnx.export` call (batch dimension only this time — every input is padded to a fixed length, unlike `05`'s growing generation context); the inference call | loading the trained torch model |

**No scipy or pandas runtime**, same reasoning as `05`.

## What's actually different from `05-transformer`

Everything *except* the mask. `numpy/layers.py` is the identical generic
layer set (duplicated here deliberately — see `CLAUDE.md`'s
self-contained-scripts convention; each model stays independently
runnable rather than importing across `models/` directories). The
softmax-backward derivation, the residual/LayerNorm wiring, Adam, the
gradient-checking tool — all the same ideas as `05`, reapplied. The one
piece worth actually sitting with is `numpy/attention.py`'s masking logic:
`05`'s causal mask is a fixed `(T, T)` pattern that's the same for every
sequence in every batch (position `i` never sees `j > i`, full stop); this
model's padding mask is `(B, 1, 1, T)` and genuinely varies *per sequence*
in the batch, because different messages have different real lengths
before the padding starts. Comparing the two `attention.py` files side by
side once you've implemented both is a good way to feel that distinction
concretely rather than just reading about it.

## Running it

```bash
# once, shared by every runtime (downloads the SMS Spam Collection):
uv run python models/06-transformer-encoder/data/make_dataset.py

# numpy: implement attention.py, verify it, then train/infer
uv run python models/06-transformer-encoder/numpy/test_gradients.py
uv run python models/06-transformer-encoder/numpy/train.py
uv run python models/06-transformer-encoder/numpy/infer.py

# torch (uses MPS on Apple Silicon, CUDA if available, else CPU):
uv run python models/06-transformer-encoder/torch/train.py
uv run python models/06-transformer-encoder/torch/infer.py

# tensorflow:
uv run python models/06-transformer-encoder/tensorflow/train.py
uv run python models/06-transformer-encoder/tensorflow/infer.py

# onnx depends on torch having been trained first:
uv run python models/06-transformer-encoder/onnx/export.py
uv run python models/06-transformer-encoder/onnx/infer.py
```

The dataset is imbalanced (~87% ham / ~13% spam), so "always predict ham"
is an ~87% baseline — expect validation accuracy to clear that
meaningfully and land in the mid-90s% within a couple thousand training
steps (verified here: both torch and tensorflow reached ~96% in just 200
steps during scaffolding verification).
