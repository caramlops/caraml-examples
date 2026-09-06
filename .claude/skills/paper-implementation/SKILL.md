---
name: paper-implementation
description: Given a downloaded research paper (PDF), scaffold a numbered models/ directory that reproduces its core experiment — using the new-model skill to build the paper's model(s) across numpy/pandas/scipy/torch/tensorflow/onnx with placeholders on the paper's actual novel contribution. Use when the user hands over a paper PDF and asks to implement, scaffold, or reproduce it.
---

# paper-implementation

Turns a downloaded paper into a `models/` entry, reusing the `new-model`
skill for the runtime scaffolding. Read `CLAUDE.md` and the `new-model`
skill first — this skill is a thin wrapper around it that adds the
paper-reading and experiment-design step.

## Steps

1. **Read the paper.** Use the `Read` tool on the PDF (use the `pages`
   param in batches for anything over ~10 pages). Extract:
   - Title, authors, venue/year, and a link/citation if available.
   - The core model/architecture the paper introduces.
   - The specific equation(s) or algorithmic step that is the paper's actual
     contribution — this is what becomes the placeholder, not incidental
     plumbing the paper also happens to use.
   - The dataset(s) used for its headline experiment.
   - The metric(s) and headline result the paper reports, so there's
     something concrete to reproduce or compare against.

2. **Scope the experiment.** Papers often report many experiments —pick the
   single smallest one that still exercises the paper's core contribution
   (usually the simplest dataset/ablation in the paper, not the SOTA table).
   State the scoping choice explicitly in `PAPER.md` (step 4) so it's clear
   what was deliberately left out and why.

3. **Pick the directory.** Same numbering scheme as `new-model` — next
   available `NN`, one sequence shared across models and papers. Slug from
   the paper's title or the technique's common name:
   `models/<NN>-<paper-slug>/`.

4. **Write `PAPER.md`** in that directory (alongside the `README.md` the
   `new-model` skill produces): title/authors/link, one-paragraph summary of
   the core idea, the key equation(s) reproduced here, which experiment from
   the paper was scoped in and why, the dataset used (and whether it's the
   paper's real dataset or a synthetic/simplified proxy — say which and
   why), and the metric/result to compare against.

5. **Dataset**: if the paper's real dataset is small and freely available,
   download it in `data/make_dataset.py`. If it's large, gated, or would
   turn this into a data-engineering exercise rather than a modeling one,
   synthesize a smaller proxy that preserves the property the paper's method
   actually exploits — record that tradeoff in `PAPER.md`.

6. **Apply the `new-model` skill's steps 4–7** for the model(s) the paper
   introduces: one `model.py`/`train.py`/`infer.py` (or `export.py`/`infer.py`
   for onnx) per runtime, with the placeholder(s) placed specifically on the
   paper's novel step — not on generic training-loop code the paper didn't
   invent. If the paper has multiple interacting components (e.g. an encoder
   and a custom loss), it's fine to have more than one placeholder per
   runtime, but call out in `PAPER.md` which placeholder corresponds to
   which contribution.

7. **Update `CHECKLIST.md`** with status `scaffolded` and type `paper`.

## What NOT to do

- Don't scaffold placeholders for well-known building blocks the paper
  merely reuses (e.g. standard attention, standard SGD) — only the paper's
  actual contribution should be a placeholder. Everything else can be fully
  implemented, even if that means using the runtime's built-in layer.
- Don't attempt the paper's full-scale experiment (large models, long
  training runs, GPU clusters) — the goal is a faithful-in-kind, small-scale
  reproduction that teaches the mechanism, not a benchmark-matching run.
