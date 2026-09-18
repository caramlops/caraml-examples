# 02 · Ridge regression

Ridge regression is OLS with an L2 penalty on the weights: minimize
`||y - Xw - b||^2 + alpha * ||w||^2` instead of just the squared error. The
intercept `b` is never penalized — only the feature weights `w` are shrunk.

**Why this needs its own model, not just an `alpha` on `01-linear-regression`**:
the dataset (`data/make_dataset.py`) is built with deliberate
multicollinearity — two of its five columns are near-linear combinations of
the others (see the module docstring). That makes `X^T X` ill-conditioned
(condition number in the low thousands), which is exactly the situation
where plain OLS's normal equation becomes numerically unstable — small
sample changes can swing fitted weights wildly on the collinear columns,
even though predictions stay reasonable. Ridge's `+ alpha * I` term keeps
the matrix comfortably invertible and pulls those spurious weights toward
zero. `TRUE_WEIGHTS = [2.0, -1.5, 1.0, 0.0, 0.0]` — the last two columns
truly contribute nothing, so a good ridge fit should push their weights
much closer to 0 than an OLS fit run on the same data would.

Every runtime below is fully wired up (data loading, training loop,
evaluation, saving/loading, a sample prediction) except for the placeholder
noted below. `ALPHA = 5.0` throughout — feel free to try other values once
things run, and watch how larger `alpha` shrinks the collinear columns'
weights harder at the cost of some bias on the real ones.

| Runtime | File(s) | Placeholder(s) | Fully implemented |
|---|---|---|---|
| numpy | `numpy/model.py` | ridge normal-equation solve (`+ alpha * I`, bias unregularized) | data load, train loop, save/load, MSE |
| pandas | `pandas/model.py` | ridge solve via centered-mean trick on a DataFrame | CSV load, MSE, save/load |
| scipy | `scipy/model.py` | ridge normal-equation solve via `scipy.linalg.solve` | fit driver, save/load, MSE |
| torch | `torch/model.py`, `torch/train.py` | `l2_penalty()`: the `alpha * sum(weights**2)` term added to the loss; the training step (`zero_grad`/`backward`/`step`) | forward pass (same as 01), data load, save/load, MSE |
| tensorflow | `tensorflow/model.py`, `tensorflow/train.py` | `l2_penalty()`, same idea as torch; the training step (`GradientTape`/`tape.gradient`/`apply_gradients`) | forward pass (same as 01), data load, save/load, MSE |
| onnx | `onnx/export.py`, `onnx/infer.py` | `torch.onnx.export` call, `onnxruntime` session run | loading the trained torch model, sample input |

Note that torch and tensorflow's `forward`/`__call__` are **not** placeholders
here — that's the matmul + bias-add you already implemented in
`01-linear-regression`. What's new to implement is the regularization term
itself (`l2_penalty()`) *and* the training step in `train.py` — see
`01-linear-regression`'s README for why the training step is a placeholder
here and not in numpy/pandas/scipy. Same story for scipy: it solves the
exact same regularized normal equation as numpy, just via
`scipy.linalg.solve` instead of `numpy.linalg.solve`.

## Alternative solution: gradient descent

`numpy/model_gd.py` + `numpy/train_gd.py` are the imperative-gradient-descent
counterpart to `numpy/model.py`'s closed-form ridge solve — same model, same
data, different training method. The one extra piece versus
`01-linear-regression`'s `numpy_gd` runtime is that the gradient itself needs
the L2 penalty's derivative added in (`2 * alpha * weights`, bias excluded),
mirroring what `l2_penalty()` contributes to the loss in torch/tensorflow.

Run it the same way as any other runtime: `uv run python
models/02-ridge-regression/numpy/train_gd.py` (after `model_gd.py`'s `_step`
placeholder is filled in). Worth comparing what you land on here against the
torch runtime's SGD result — same loss, same penalty, computed by hand vs.
via autograd.

## Alternative solutions: framework abstraction levels

Same idea as `01-linear-regression`'s abstraction-level spectrum, extended
to ridge's L2 penalty:

| File | Level | Where the L2 penalty lives |
|---|---|---|
| `torch/model_raw.py` | lowest | added into `loss` by hand in `fit()`, same as `l2_penalty()` elsewhere |
| `torch/model.py` | middle | `l2_penalty()` method, added to the loss in `train.py` |
| `torch/model_layer.py` | highest | `l2_penalty()` method reading `nn.Linear`'s `.weight`, added to the loss in `train.py` |
| `tensorflow/model.py` | lowest | added into `loss` by hand inside the `GradientTape` block |
| `tensorflow/model_keras.py` | highest | declared on the layer itself via `kernel_regularizer=tf.keras.regularizers.l2(alpha)` |

The Keras version is the interesting contrast here, more than just "less
code": you don't add the penalty to the loss yourself at all.
`kernel_regularizer` registers it as a property of the layer, and Keras
folds every layer's registered regularization losses into the total loss
automatically during `model.fit()`/`model.evaluate()`. That's a genuinely
different mental model from every other runtime here (penalty as something
you compute vs. penalty as something you declare), not just a shorter way
to write the same thing.

Two scaffolding details worth knowing about (already handled for you, not
part of the placeholder): `model_keras.py` explicitly zero-initializes the
Dense layer (Keras defaults to random Glorot-uniform init, unlike every
other runtime's zero start), and `train_keras.py` passes
`batch_size=X_train.shape[0]` to `model.fit()` (Keras defaults to
mini-batch SGD with `batch_size=32`, which would take many gradient steps
per "epoch" instead of the one full-batch step every other runtime takes).
Both were necessary to get `model_keras.py` to converge to the *exact* same
loss trajectory as `torch/model_raw.py` — worth remembering next time a
Keras model behaves differently than expected for no obvious reason.

`torch/train_layer.py`, `tensorflow/train_keras.py`'s training step is also
a placeholder, same as the plain `torch`/`tensorflow` runtimes above —
`train_layer.py`'s is the `zero_grad`/`backward`/`step` sequence (with
`model.l2_penalty(ALPHA)` folded into the loss), `train_keras.py`'s is
`model.compile(...)` + `model.fit(...)` (the penalty is already baked into
the model via `kernel_regularizer`, so there's nothing extra to add here).
`train_raw.py` stays fully implemented — its training step already lives in
`model_raw.py`'s `_step()`.

Run them like any other runtime, e.g. `uv run python
models/02-ridge-regression/torch/train_raw.py`, `train_layer.py`, or
`models/02-ridge-regression/tensorflow/train_keras.py`.

## Running it

```bash
# once, shared by every runtime:
uv run python models/02-ridge-regression/data/make_dataset.py

# any runtime, e.g. numpy:
uv run python models/02-ridge-regression/numpy/train.py
uv run python models/02-ridge-regression/numpy/infer.py

# onnx depends on torch having been trained first:
uv run python models/02-ridge-regression/torch/train.py
uv run python models/02-ridge-regression/onnx/export.py
uv run python models/02-ridge-regression/onnx/infer.py
```

Run with `CARAML_LOGLEVEL=DEBUG` to see intermediate shapes while you work
on a placeholder.
