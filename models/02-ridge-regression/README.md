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

| Runtime | File | Placeholder | Fully implemented |
|---|---|---|---|
| numpy | `numpy/model.py` | ridge normal-equation solve (`+ alpha * I`, bias unregularized) | data load, train loop, save/load, MSE |
| pandas | `pandas/model.py` | ridge solve via centered-mean trick on a DataFrame | CSV load, MSE, save/load |
| scipy | `scipy/model.py` | residual function augmented with `sqrt(alpha) * weights` | fit driver, save/load, MSE |
| torch | `torch/model.py` | `l2_penalty()`: the `alpha * sum(weights**2)` term added to the loss | forward pass (same as 01), SGD training loop, save/load |
| tensorflow | `tensorflow/model.py` | `l2_penalty()`, same idea as torch | forward pass (same as 01), `GradientTape` training loop, save/load |
| onnx | `onnx/export.py`, `onnx/infer.py` | `torch.onnx.export` call, `onnxruntime` session run | loading the trained torch model, sample input |

Note that torch and tensorflow's `forward`/`__call__` are **not** placeholders
here — that's the matmul + bias-add you already implemented in
`01-linear-regression`. The new thing to implement is the regularization
term itself, in `l2_penalty()`.

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
