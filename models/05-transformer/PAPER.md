# Attention Is All You Need / GPT (decoder-only transformer)

**Core mechanism**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. "Attention Is
All You Need." NeurIPS 2017. https://arxiv.org/abs/1706.03762

**Decoder-only framing**: Alec Radford, Karthik Narasimhan, Tim Salimans,
Ilya Sutskever. "Improving Language Understanding by Generative
Pre-Training" (GPT). 2018.
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

## The core idea

Before this paper, sequence models (RNNs, LSTMs) processed tokens one at a
time, carrying state forward step by step — inherently sequential, and
prone to losing information over long distances. "Attention Is All You
Need" replaces recurrence entirely with **self-attention**: every position
in a sequence directly computes a weighted combination of *every other
position*, in parallel, with the weights themselves learned from the data.
GPT then applies this architecture to a single, simple objective —
autoregressive next-token prediction — and shows that scaling it up (more
layers, more data) produces a general-purpose language model.

## Key equations reproduced here

**Scaled dot-product attention** (eq. 1):

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**Multi-head attention** (eq. 2): run `h` attention computations in
parallel on learned linear projections of `Q`/`K`/`V`, then concatenate
and project the result:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

**Sinusoidal positional encoding** (eq. 3/4) — since self-attention has no
inherent notion of token order (unlike an RNN), position has to be injected
explicitly:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Causal masking** (decoder self-attention, section 3.2.3): position `i`
may only attend to positions `<= i`, implemented by setting the
pre-softmax score for every `j > i` to `-inf` so it receives exactly zero
attention weight.

**Position-wise feedforward** (eq. 2 of section 3.3): a two-layer MLP
applied identically (same weights) at every position:

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

## What's a deliberate deviation from the paper

- **Pre-LN, not post-LN.** The original paper applies LayerNorm *after*
  each residual addition (`LayerNorm(x + Sublayer(x))`). This
  implementation applies it *before* each sub-layer instead
  (`x + Sublayer(LayerNorm(x))`) — what GPT-2 and effectively every modern
  decoder-only transformer actually uses, because it trains more stably
  without a learning-rate warmup schedule. Since the whole point here is
  training a small model and actually seeing it learn, pre-LN was the
  right practical choice, not just a stylistic one.
- **Decoder-only, not encoder-decoder.** The original paper's experiments
  are a full encoder-decoder machine translation model. This reproduces
  only the decoder stack (self-attention + feedforward, no
  cross-attention to an encoder) — the half GPT builds on, and the half
  relevant to the "generate text" deliverable here.

## Scoped-down experiment

The paper trains on large machine-translation datasets (WMT), for which
GPT's successors train on massive web-scraped text corpora — both entirely
impractical to reproduce locally. This instead trains a **small**
(~1-3M parameter, 3-layer, 4-head, 64-dim) character-level language model
on the **tiny-Shakespeare** corpus (~1MB, ~1M characters) — the same
dataset and tokenization scheme Karpathy's char-rnn/nanoGPT tutorials use.
It's real, structured text (not synthetic data — see `data/make_dataset.py`
for why that matters here), small enough to train from scratch in minutes
on a laptop (even without CUDA — this repo's numpy/torch runtimes were
verified on an Apple M2's MPS backend), while still exercising every part
of the real mechanism: multi-head causal self-attention, positional
encoding, residual connections, and autoregressive generation.

## Metric / result to compare against

There's no benchmark number to match here (tiny-Shakespeare char-LMs don't
have a standard reported metric the way, say, ImageNet classification
does) — the deliverable is qualitative: generated text that's recognizably
*trying* to be Shakespearean dialogue (character names, stage-direction-like
structure, archaic word choices) after a few thousand training steps,
even if not grammatically coherent at this model size. Training/validation
cross-entropy loss should drop from `~ln(vocab_size) ≈ 4.17` (random-guess
baseline for this 65-character vocabulary) down into the `1.5-2.0` range.
