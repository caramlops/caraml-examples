# 05 · Transformer (decoder-only, GPT-style)

A small autoregressive, character-level language model: predict the next
character given the previous ones, via multi-head causal self-attention.
See `PAPER.md` for the papers this reproduces (Vaswani et al.'s attention
mechanism, GPT's decoder-only framing), the exact equations, and what was
deliberately scoped down or deviated from.

The data is the real tiny-Shakespeare corpus (`data/make_dataset.py`
downloads it once), not synthetic — a from-scratch language model trained
on synthetic gibberish wouldn't tell you anything about whether your
implementation actually learned language structure.

| Runtime | File(s) | Placeholder(s) | Fully implemented |
|---|---|---|---|
| numpy | `numpy/attention.py` | multi-head causal self-attention, **forward and backward both** (no autograd in this runtime) | `numpy/layers.py` (every other layer + Adam), `numpy/model.py` (architecture wiring), training loop, generation, `numpy/test_gradients.py` (gradient checker) |
| torch | `torch/model.py`, `torch/train.py` | attention forward pass (autograd handles backward); the training step (`zero_grad`/`backward`/`step`) | architecture wiring, data load, save/load, generation |
| tensorflow | `tensorflow/model.py`, `tensorflow/train.py` | attention forward pass; the training step (`GradientTape`/`tape.gradient`/`apply_gradients`) | architecture wiring, data load, save/load, generation |
| onnx | `onnx/export.py`, `onnx/infer.py` | `torch.onnx.export` call (two dynamic dims this time — batch *and* sequence length); the generation loop | loading the trained torch model |

**No scipy or pandas runtime for this model.** Neither has anything
distinct to practice here: there's no closed form (scipy's `01`/`02` role)
or generic-optimizer story (scipy's `03` role) for a transformer, and
there's no DataFrame-native derivation of attention for pandas to offer —
both would just be ceremony around the same numpy/torch code.

## The big one: `numpy` has real backward passes, not just forward

Every other model in this repo either has a closed form (`01`-`02`) or
lets autograd handle backpropagation (`03`-`04`, and this model's torch/
tensorflow runtimes). `numpy/layers.py` and `numpy/model.py` implement a
tiny manual-autograd convention instead: every layer has a `forward()`
that returns `(output, cache)` and a `backward(dout, cache)` that returns
gradients for every input, including parameters — composed by hand into a
full forward and backward pass through the whole transformer stack. This
is fully implemented for every layer *except* attention
(`numpy/attention.py`) — that one's forward **and** backward are both the
placeholder, because it's the paper's actual contribution; every other
layer here is a well-known building block the paper reuses.

**`numpy/test_gradients.py`** is the tool to trust your attention backward
derivation rather than hoping it's right: it numerically estimates
`d(loss)/d(param)` by finite differences on a deliberately tiny random
model instance, and compares that against your `backward()`'s analytic
gradient for *every* parameter in the model. Run it after implementing
`attention.py`:

```bash
uv run python models/05-transformer/numpy/test_gradients.py
```

A correct implementation gets relative error well under `1e-4` on every
parameter; a real bug in the derivation or the code shows up as a specific
named parameter failing (almost always one of `attn.Wq`/`Wk`/`Wv`/`Wo`,
since every other layer here is already implemented and was verified this
same way before being shipped).

## Running it

```bash
# once, shared by every runtime (downloads tiny-Shakespeare):
uv run python models/05-transformer/data/make_dataset.py

# numpy: implement attention.py, verify it, then train/generate
uv run python models/05-transformer/numpy/test_gradients.py
uv run python models/05-transformer/numpy/train.py
uv run python models/05-transformer/numpy/infer.py

# torch (uses MPS on Apple Silicon, CUDA if available, else CPU):
uv run python models/05-transformer/torch/train.py
uv run python models/05-transformer/torch/infer.py

# tensorflow:
uv run python models/05-transformer/tensorflow/train.py
uv run python models/05-transformer/tensorflow/infer.py

# onnx depends on torch having been trained first:
uv run python models/05-transformer/onnx/export.py
uv run python models/05-transformer/onnx/infer.py
```

Training logs train/val loss every 200 steps — expect it to drop from
around `ln(65) ≈ 4.17` (random-guess baseline for this 65-character
vocabulary) into the `1.5-2.0` range over a few thousand steps. Generated
text won't be grammatical at this model size, but should start looking
recognizably Shakespeare-*shaped*: character names, dialogue structure,
archaic diction.
