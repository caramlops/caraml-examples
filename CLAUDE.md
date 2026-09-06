# caraml-examples

## Purpose

This repo has two goals that reinforce each other:

1. **Learn ML fundamentals** by implementing models and papers from a personal
   learning checklist (see `CHECKLIST.md`), from scratch, in the runtimes a
   working ML researcher actually touches day to day.
2. **Build a corpus** of comparable implementations of the same model across
   those runtimes, which will later feed MLops tooling (`caraml`) — e.g.
   tooling that needs to reason about a model regardless of which runtime it
   was written in.

## Layout

```
models/
  01-linear-regression/
    README.md              # what the model is, the math, what's a placeholder
    data/
      make_dataset.py       # fully implemented: generates/splits data, no placeholders
    numpy/
      model.py              # placeholder: the actual solver/algorithm
      train.py               # fully implemented: load data, fit, evaluate, save
      infer.py                # fully implemented: load saved model, predict
    pandas/   {model.py, train.py, infer.py}
    scipy/    {model.py, train.py, infer.py}
    torch/    {model.py, train.py, infer.py}
    tensorflow/ {model.py, train.py, infer.py}
    onnx/     {export.py, infer.py}   # exports a trained torch model, runs it via onnxruntime
  02-<next-model>/
    ...
```

Each numbered directory is one model or paper. Numbering follows learning
order from `CHECKLIST.md`, not difficulty or category.

Everything lives under `models/` — there are no standalone scratch scripts
at the repo root. Datasets and trained model artifacts are regenerable
(`.gitignore`d) and never need to be preserved.

## Conventions

- **Runtimes**: numpy, pandas, scipy, torch, tensorflow, onnx. Not every
  runtime makes sense for every model (e.g. onnx is inference/export only,
  not training) — use judgment, but default to all six.
- **Placeholders**: the data pipeline, training loop, saving/loading, and
  metric reporting are always fully implemented and runnable. Only the
  *significant piece* — the actual algorithm (an OLS solver, a gradient step,
  a forward pass's tensor ops) — is left as a placeholder that raises
  `NotImplementedError`, with a comment describing exactly what to implement
  and the shapes involved. The goal is that filling in one function makes the
  whole script work end-to-end.
- **Self-contained scripts**: each runtime's `train.py`/`infer.py` is run
  directly (`uv run python models/01-linear-regression/numpy/train.py`), not
  imported as a package (directory names have digits/hyphens and aren't
  valid module names). They import sibling `model.py` via a
  `sys.path.insert(0, str(Path(__file__).parent))` shim — copy that pattern.
- **Shared data**: data generation/download and train/test split live once,
  in `data/make_dataset.py`, and are saved as both `.npz` (numpy/scipy/
  torch/tensorflow) and `.csv` (pandas) so every runtime reads the same
  split without re-implementing it.
- Prefer synthetic, seeded data (reproducible, no network dependency) unless
  a real dataset materially matters to what's being learned.

## Skills

- `.claude/skills/new-model` — scaffolds a new model across all runtimes
  following the conventions above. Use this to add the next checklist item.
- `.claude/skills/paper-implementation` — given a downloaded paper PDF,
  scaffolds an experiment reproduction, using the `new-model` skill for the
  model(s) the paper introduces.

## Checklist

`CHECKLIST.md` tracks the queue of models/papers to implement and their
status. Update it when scaffolding or completing an item.
