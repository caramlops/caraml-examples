# BERT (encoder-only transformer)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. "BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding." NAACL 2019. https://arxiv.org/abs/1810.04805

Builds on the encoder half of Ashish Vaswani et al., "Attention Is All You
Need" (NeurIPS 2017), the same paper `05-transformer`'s decoder half
reproduces — see that model's `PAPER.md` for the shared attention-mechanism
equations.

## The core idea

`05-transformer` reproduces the *decoder* side of the original transformer:
causal self-attention, where position `i` can only see positions `<= i`,
which is exactly what you need for autoregressive generation (predicting
the next token from only what came before it). BERT's contribution is
showing that the *encoder* side — **bidirectional** self-attention, where
every position sees every other position, with no causal restriction at
all — is a far better representation for *understanding* a sequence
(classification, not generation): a word's meaning genuinely depends on
context from both directions ("bank" means something different depending
on whether the next word is "account" or "river").

The second half of BERT's contribution — the `[CLS]` token — is an
architectural trick for turning a sequence of per-token representations
into one fixed-size vector for a downstream task: prepend a special token
to every input, and use *its* final-layer representation (after it's
attended to everything else in the sequence) as a pooled summary for
classification.

## Key equations/mechanisms reproduced here

- **Bidirectional multi-head self-attention** — identical scaled
  dot-product mechanics to `05-transformer`'s (`Attention(Q,K,V) =
  softmax(QK^T / sqrt(d_k)) V`, split across heads), but masking based on
  *padding* (which tokens are real vs. filler) rather than *position*
  (which tokens come before/after). See
  `numpy/attention.py` for exactly how that mask differs from `05`'s.
- **`[CLS]`-token pooling for classification**: `model.py`'s final step
  reads only `x[:, 0, :]` (the `[CLS]` position's representation after the
  full encoder stack) and feeds it to a linear classification head —
  BERT's actual mechanism for adapting a sequence encoder to a
  fixed-output task.
- **Sinusoidal positional encoding**, same equations as `05`.

## What's a deliberate deviation from the paper

- **No masked-language-model / next-sentence-prediction pretraining.**
  BERT's actual training recipe is unsupervised pretraining (mask random
  tokens, predict them; predict whether two sentences are adjacent) on a
  massive corpus, *then* fine-tuning on a small labeled dataset. That
  two-stage recipe is a large undertaking on its own and would mostly
  duplicate `05-transformer`'s language-modeling objective rather than
  teaching something new architecturally. This reproduces BERT's
  *architecture* (bidirectional attention + `[CLS]` pooling) trained
  directly, supervised, on the downstream classification task — the part
  of the paper that's actually novel relative to `05`.
- **Character-level, not WordPiece.** The paper uses a learned subword
  vocabulary; this uses the same character-level tokenization as `05`, to
  keep the two models' infrastructure directly comparable and avoid a
  second, unrelated tokenization-algorithm detour.

## Scoped-down experiment

The paper pretrains on BooksCorpus + Wikipedia (several GB) before
fine-tuning. This instead trains a small (~1-3M parameter, 3-layer,
4-head, 64-dim) encoder directly on the **SMS Spam Collection** dataset —
5,574 real text messages labeled `ham`/`spam`, a classic, tiny (~500KB),
freely downloadable binary text-classification dataset. Real, labeled
text (not synthetic) is used deliberately: a classifier trained on
synthetic text-label pairs wouldn't test whether the architecture can
actually learn a genuine text-classification signal.

## Metric / result to compare against

Binary classification accuracy on the held-out validation split. The
dataset is imbalanced (~87% ham / ~13% spam), so the baseline to beat is
"always predict ham" (~87% accuracy) — a working implementation should
clear that baseline meaningfully and reach somewhere in the mid-90s%
within a couple thousand training steps.
