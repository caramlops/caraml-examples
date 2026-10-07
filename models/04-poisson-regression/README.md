# 04 · Poisson regression (GLM, IRLS)

Count-data regression: given `x`, predict the mean of a Poisson-distributed
count `y` via `mu = exp(x @ weights + bias)` (the "log link"), fit by
maximizing the Poisson log-likelihood. The data is synthetic and seeded
(`data/make_dataset.py`): `y ~ Poisson(exp(X @ [0.4, 0.3, -0.5] + 1.0))`.

This is this repo's first **generalized linear model (GLM)** — `01`/`02`
(Gaussian outcome, identity link) and `03` (categorical outcome, softmax
link) are GLMs too, just with a different outcome distribution and link
function. The actual new thing to learn here isn't "yet another regression,"
it's the **fitting algorithm**: **IRLS** (iteratively reweighted least
squares), a Newton's-method-flavored technique that's different from both
`01`/`02`'s one-shot closed form and `03`'s plain gradient descent.

**Why plain OLS doesn't work on count data**: OLS implicitly assumes
constant variance across all predictions. A true Poisson variable has
`Var(Y) = E[Y]` — its variance scales *with* its own mean, so a count near 0
is far more tightly constrained than a count near 50. Plain least squares
has no way to express that, and would weight every sample equally regardless
of how much the model should trust it.

**What IRLS actually does, concretely**: each iteration solves a *weighted*
version of `01-linear-regression`'s exact normal equation —
`(X^T @ diag(W) @ X) @ w = X^T @ diag(W) @ z` — where `z` is a "working
response" (a linearized pseudo-target built from the current residual) and
`W` is a per-sample weight reflecting how much the current fit trusts that
sample. Solve, recompute `z`/`W` from the new fit, solve again. For Poisson
regression with the log link (its *canonical* link), both `z` and `W`
collapse to remarkably clean formulas — the same kind of clean simplification
that made softmax + cross-entropy's combined gradient so simple in `03`.
IRLS typically converges in single-digit iterations, not the hundreds/
thousands gradient descent needs, because it's using second-order
(curvature) information the way Newton's method does, not just the
gradient's direction.

| Runtime | File(s) | Placeholder(s) | Fully implemented |
|---|---|---|---|
| numpy | `numpy/model.py` | Poisson/log-link working quantities (`mu`, `W`, `z`); the weighted normal-equation IRLS step | data load, train loop, save/load, MSE/deviance |
| scipy | `scipy/model.py` | same two, with the weighted solve via `scipy.linalg.solve` | fit driver, save/load, MSE/deviance |
| torch | `torch/model.py`, `torch/train.py` | forward pass (raw linear predictor, no `exp()`); the training step (`zero_grad`/`backward`/`step`) | data load, save/load, MSE |
| tensorflow | `tensorflow/model.py`, `tensorflow/train.py` | forward pass (raw linear predictor); the training step (`GradientTape`/`tape.gradient`/`apply_gradients`) | data load, save/load, MSE |
| onnx | `onnx/export.py`, `onnx/infer.py` | `torch.onnx.export` call, `onnxruntime` session run | loading the trained torch model, sample input |

**No pandas runtime**, same reasoning as `03`: IRLS has no alternative
DataFrame-native derivation to practice, so a pandas version would just be
the identical algorithm with extra ceremony around reading columns.

scipy's role here is the `01`/`02`-style `scipy.linalg` idiom again (solving
the per-iteration weighted normal equation via `scipy.linalg.solve`), not
`03`'s `scipy.optimize` generic-optimizer story — that story was already
told once, so it isn't repeated here just because this model also lacks a
single-shot closed form.

## The one gotcha specific to this model: torch and tensorflow expect `log(mean)`, not `mean`

`torch/model.py` and `tensorflow/model.py`'s forward pass returns the raw
linear predictor `eta` (mathematically `log(mu)`), not the predicted mean
count itself — deliberately, and for the same reason `03`'s forward pass
returns logits instead of probabilities. `nn.PoissonNLLLoss` (torch,
`log_input=True` by default) and `tf.nn.log_poisson_loss` (tensorflow) both
expect `log(mean)` directly and apply `exp()` internally as part of a single
numerically stable loss computation. `infer.py` and `train.py`'s evaluation
code apply `exp()`/`torch.exp()`/`tf.exp()` themselves afterward, the same
division of labor as `03`'s softmax-after-the-loss pattern.

## Running it

```bash
# once, shared by every runtime:
uv run python models/04-poisson-regression/data/make_dataset.py

# any runtime, e.g. numpy:
uv run python models/04-poisson-regression/numpy/train.py
uv run python models/04-poisson-regression/numpy/infer.py

# onnx depends on torch having been trained first:
uv run python models/04-poisson-regression/torch/train.py
uv run python models/04-poisson-regression/onnx/export.py
uv run python models/04-poisson-regression/onnx/infer.py
```

Each runtime's `train.py` prints test-set MSE (numpy/scipy also print mean
Poisson deviance, the more statistically appropriate metric for count data)
— since every runtime fits the same data to the same log-likelihood, once
implemented they should all converge to the same weights: Poisson
regression's log-likelihood is globally concave, so IRLS (Newton's method)
and gradient descent both find the *same* unique optimum, not just similar
ones.
