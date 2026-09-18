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

   **scipy's role**: when the model has a closed-form solution, scipy
   solves the *same* closed-form equation as numpy/pandas — just via
   `scipy.linalg` instead of `numpy.linalg` (e.g. `scipy.linalg.solve(...,
   assume_a="pos")` for a symmetric positive-definite normal equation). It
   is not a vehicle for demonstrating iterative/gradient-based optimization
   — see "Alternative solutions" below for where that practice belongs.
   Reach for `scipy.optimize.least_squares` instead of `scipy.linalg` only
   for models that are genuinely nonlinear in their parameters (no normal
   equation exists at all) — that's a different, rarer case, not the
   default for anything with a closed form.

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
   - Log a short summary (row counts, paths written) when run — see step 4
     for the logging pattern.

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
     logs results, and saves the fitted parameters next to itself
     (`model.npz`/`model.pt`/`model.json`, whatever fits the runtime).
     Fully implemented — this file must run correctly once the `model.py`
     placeholder is filled in, with no further edits needed.
     - **Exception for torch and tensorflow**: the training loop body
       itself is a second placeholder here, not fully implemented — set up
       the model, optimizer/loss, and data (all plumbing), then leave the
       actual step as `raise NotImplementedError(...)` with a comment
       spelling out the sequence:
       - torch: `optimizer.zero_grad()` → `preds = model(X_train)` →
         `loss = loss_fn(...)` → `loss.backward()` → `optimizer.step()`.
       - tensorflow: `with tf.GradientTape() as tape:` (compute preds/loss
         inside it) → `grads = tape.gradient(loss, model.trainable_variables)`
         → `optimizer.apply_gradients(zip(grads, model.trainable_variables))`.
       - Keras (`model_keras.py`'s `train_keras.py`, if present): the
         placeholder is `model.compile(...)` + `model.fit(...)` itself —
         that pairing *is* the significant Keras idiom being practiced,
         contrasted against the other two runtimes writing the loop by
         hand. Keep the zero-init / `batch_size=X_train.shape[0]`
         scaffolding notes in the comment regardless (see the abstraction-
         levels section below) — those are plumbing gotchas, not the thing
         being taught.
       Reason: for these two runtimes, the autograd/GradientTape training
       mechanics are as central to what's being learned as the forward
       pass — unlike numpy/pandas/scipy, where `train.py` really is just
       boilerplate around a one-shot closed-form solve. Don't add this to
       `torch/model_raw.py`'s `train_raw.py` — `model_raw.py`'s `_step()`
       already covers the same ground for that alternative.
   - `infer.py`: loads the saved parameters, runs one example prediction,
     logs it. Fully implemented.
   - Import pattern for pulling in sibling `model.py`:
     ```python
     from pathlib import Path
     import sys
     sys.path.insert(0, str(Path(__file__).parent))
     from model import <ModelClass>
     ```
   - Logging: `model.py` just does `import logging` + `log =
     logging.getLogger(__name__)` (no config — it's a library). `train.py`/
     `infer.py` additionally call `logging.basicConfig(level=os.environ.get(
     "CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")`
     once, near the top. Use `log.info(...)` wherever you would have
     `print(...)`ed a result (rows loaded, epoch loss, test metric, save
     path), and `log.debug(f"X shape={X.shape}")`-style calls (f-strings,
     not `%s`-style) after loading data / around the placeholder's core
     operation, so filling in the placeholder comes with
     `CARAML_LOGLEVEL=DEBUG` shape visibility for free. See `CLAUDE.md`'s
     Debugging/logging convention.

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

8. **Format**: run `uv run black .` before finishing.

## Alternative solutions (optional): practicing training methods

For models with a closed-form solve, it's often worth also practicing
*how* you'd fit the same model without one — imperative gradient descent by
hand, as the numpy-side counterpart to torch/tensorflow's autograd-based
training loop. When this is pedagogically relevant for the model:

- Add extra files **inside the existing numpy runtime folder** — do not
  create a new top-level runtime directory. Name them `model_gd.py`
  (the model class, with a `_step(self, X, residuals)` method as the one
  placeholder — compute the loss gradient and update parameters in place)
  and `train_gd.py` (same structure as `train.py`, saving to a distinct
  `model_gd.npz` so it doesn't clobber the closed-form model's artifact).
- This is a different *training method* for the same model, not a
  different library idiom — that's why it lives alongside `numpy/model.py`
  rather than getting its own runtime directory.
- Document it in the model's README under its own "Alternative solution"
  heading, and explicitly point out that torch's existing SGD training loop
  is the computation-graph-based counterpart — the pairing of `numpy_gd`
  (manual gradients) and `torch` (autograd) is the point, not a full
  separate exploration of every optimizer.
- Don't add this by default to every model — only when the model actually
  benefits from contrasting a closed-form solve against an iterative one
  (e.g. linear/ridge regression). Skip it for models that don't have a
  closed form to contrast against in the first place.

## Alternative solutions (optional): framework abstraction levels

torch and tensorflow both span a range from "raw computation graph" to
"pre-built layer," and it's worth letting the user practice more than one
rung of it when the model is simple enough that the layer-construction
placeholder still teaches something (a single Dense/Linear layer, not a
whole architecture). Same rule as gradient descent above: extra files
inside the existing `torch/`/`tensorflow/` folders, not new top-level
runtimes.

- **torch**: `model.py` (the default) uses `nn.Module` + hand-declared
  `nn.Parameter`s — a middle rung. Add `model_layer.py` using `nn.Linear`
  (or the relevant built-in layer) as the high rung, and/or `model_raw.py`
  using plain `torch.Tensor(requires_grad=True)` with no `nn.Module` at all
  as the low rung. `model_raw.py`'s placeholder belongs in a `_step()`
  method (manual `loss.backward()` + `torch.no_grad()` update + manual
  `.grad.zero_()`), not the forward pass — see `01-linear-regression`'s
  `torch/model_raw.py` for the worked pattern. This is also the natural
  successor to the gradient-descent alternative above: numpy_gd computes
  gradients by hand and updates by hand; `model_raw.py` lets autograd
  compute gradients but still updates by hand; `model.py`/`model_layer.py`
  hand both jobs to `torch.optim`.
- **tensorflow**: `model.py` (the default) uses `tf.Module` + hand-declared
  `tf.Variable`s — the low rung already. Add `model_keras.py` (a
  `build_model(n_features)` function, not a class — match Keras's own
  idiom) using `tf.keras.layers.Dense` as the high rung. Its `train_keras.py`
  should use `model.compile(...)` + `model.fit(...)`, not a manual
  `GradientTape` loop — that's the actual point of the Keras rung. Use an
  explicit `tf.keras.layers.Input(shape=(n_features,))` for the input shape
  rather than `Dense`'s deprecated `input_shape=` kwarg. Two defaults Keras
  gets wrong for this repo's purposes — set both explicitly, as scaffolding,
  not placeholders:
  - `Dense(..., kernel_initializer="zeros", bias_initializer="zeros")` —
    Keras defaults to random Glorot-uniform init, but every other runtime
    here starts from zero, so leaving it random breaks comparability
    between runtimes' results.
  - `model.fit(..., batch_size=X_train.shape[0])` — `fit()` defaults to
    `batch_size=32` (mini-batch SGD across many steps per epoch), while
    every other runtime does full-batch gradient descent (one step per
    epoch over the whole training set). Without this, epoch counts and
    loss trajectories won't line up with the other runtimes at all.
  - If the model has a regularization term (e.g. ridge's L2 penalty),
    prefer expressing it declaratively where Keras supports it (e.g.
    `kernel_regularizer=tf.keras.regularizers.l2(alpha)`) rather than
    manually adding it to the loss — that contrast (penalty as something
    you declare vs. something you compute) is itself worth teaching, not
    just incidental. But if you do this, `model.evaluate()`'s reported loss
    will include the regularization term too — compute a plain metric by
    hand (`model.predict(...)` then the raw MSE) if you need a number
    comparable to the other runtimes' unpenalized test metric.
- For layer-based files, the placeholder is model *construction*
  (instantiating the layer), not the forward pass — a pre-built layer owns
  its own weights/bias internally, so there's nothing left to hand-implement
  in `forward`/`__call__` once it exists.
- Skip this for models complex enough that "the layer API version" would
  mean building a real multi-layer architecture rather than swapping one
  line — at that point it stops teaching the abstraction-level contrast and
  just becomes a second full implementation to maintain.

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
