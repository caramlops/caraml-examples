---
name: new-model
description: Scaffold a new ML model as a numbered directory under models/, implemented across numpy, pandas, scipy, torch, tensorflow, and onnx, with the data pipeline and training/inference plumbing fully working and only the significant algorithmic piece left as a placeholder. Use when the user asks to add a new model to caraml-examples, implement <some algorithm> from scratch, or work through the next item on their model-learning checklist.
---

# new-model

Scaffolds one model, comparably implemented across runtimes, following the
conventions in the repo's `CLAUDE.md`. Read that file first if you haven't
already — it defines the directory layout and placeholder conventions this
skill implements.

**Reference implementation**: `models/01-linear-regression/` is a complete,
worked example of this skill's output. When in doubt about a pattern (import
shim, file split, how much to leave as a placeholder), copy what it does.

## Steps

1. **Pick the directory number and name.** List `models/`, find the highest
   `NN` prefix, and use `NN+1` (zero-padded to 2 digits). Slugify the model
   name in kebab-case: `models/<NN>-<model-slug>/`.

2. **Decide which runtimes apply.** Default to all six: numpy, pandas,
   scipy, torch, tensorflow, onnx. Drop one only if it's genuinely
   nonsensical for this model (rare) — note the omission and why in the
   model's README. Remember onnx is inference/export-only: it converts a
   trained model (default to exporting from the torch implementation) and
   runs it via `onnxruntime`, it doesn't train.

3. **Write `data/make_dataset.py` — fully implemented, no placeholders.**
   - Generate (preferred: synthetic, numpy `default_rng` with a fixed seed)
     or download the data appropriate to the model. Only download real data
     if the model's value is specifically about handling real-world data
     characteristics — otherwise synthetic + seeded is more reproducible and
     removes network flakiness from the learning loop.
   - Do the train/test split here, once.
   - Save the result as `data/<slug>.npz` (arrays: `X_train`, `y_train`,
     `X_test`, `y_test`, or whatever's appropriate to the model) AND as
     `data/train.csv` / `data/test.csv` for the pandas runtime to read
     directly with `pd.read_csv`.
   - Print a short summary (row counts, paths written) when run.

4. **For each runtime, write three files** (two for onnx: `export.py`,
   `infer.py`):
   - `model.py`: the model as a class. The constructor and `predict`/
     `forward`/`__call__` plumbing is implemented; the **one method that
     embodies the model's core math** (the solver, the gradient step, the
     forward pass's tensor ops) raises `NotImplementedError` with a comment
     that:
     - names the exact operation(s) to write (e.g. "matrix-multiply x by
       self.weights, then add self.bias"),
     - states the shapes involved,
     - gives enough of the formula that the user is implementing the
       concept, not reverse-engineering the file structure.
     Keep exactly one placeholder per runtime unless the model genuinely has
     two independent significant pieces (e.g. a paper with a custom loss
     *and* a custom layer) — more than that dilutes what's being practiced.
   - `train.py`: loads data via the shim (see below), builds the model,
     fits/trains it, evaluates on the test split with a sensible metric,
     prints results, and saves the fitted parameters next to itself
     (`model.npz`/`model.pt`/`model.json`, whatever fits the runtime).
     Fully implemented — this file must run correctly once the `model.py`
     placeholder is filled in, with no further edits needed.
   - `infer.py`: loads the saved parameters, runs one example prediction,
     prints it. Fully implemented.
   - Import pattern for pulling in sibling `model.py`:
     ```python
     from pathlib import Path
     import sys
     sys.path.insert(0, str(Path(__file__).parent))
     from model import <ModelClass>
     ```

5. **Write the model's `README.md`** at `models/<NN>-<slug>/README.md`:
   what the model is, the core formula/algorithm in plain terms, which
   runtimes implement it, exactly what's a placeholder in each vs. fully
   implemented, and how to run each (`uv run python models/<NN>-<slug>/<runtime>/train.py`
   then `infer.py`, noting onnx depends on torch having been trained first).
   Link out to any paper/reference the model comes from if relevant.

6. **Update `CHECKLIST.md`** — add or update the row for this model with
   status `scaffolded`.

7. **Dependencies**: if a runtime needs a package not yet in `pyproject.toml`
   (pandas, scipy, tensorflow, onnx, onnxruntime, etc.), add it under
   `[project.dependencies]` and tell the user to run `uv lock && uv sync`
   — don't run heavy installs yourself without asking.

## What NOT to do

- Don't pre-solve the placeholder "just to make sure it works" and then
  comment it out — leave it as `raise NotImplementedError(...)`. The point
  is for the user to write it.
- Don't over-abstract across runtimes with a shared base class or shared
  training-loop utility — the repo's value is that each runtime's
  implementation is self-contained and idiomatic to that library, even if
  that means some duplication.
- Don't add a `requirements.txt` or per-model virtualenv — this repo uses
  one `uv`-managed environment for everything under it.
