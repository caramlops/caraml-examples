# 01 · Linear regression

Ordinary least squares: fit `y = X @ weights + bias` to minimize squared
error. The data is synthetic and seeded (`data/make_dataset.py`) so results
are reproducible across runtimes: `y = X @ [3.0, -2.0, 0.5] + 5.0 + noise`.

Every runtime below is fully wired up (data loading, training loop,
evaluation, saving/loading, a sample prediction) except for **one placeholder
per runtime** — the actual mathematical step. Fill it in, then run
`train.py` followed by `infer.py`.

| Runtime | File(s) | Placeholder(s) | Fully implemented |
|---|---|---|---|
| numpy | `numpy/model.py` | OLS normal-equation solve | data load, train loop, save/load, MSE |
| pandas | `pandas/model.py` | OLS solve via centered-mean trick on a DataFrame | CSV load, MSE, save/load |
| scipy | `scipy/model.py` | OLS normal-equation solve via `scipy.linalg.solve` | fit driver, save/load, MSE |
| torch | `torch/model.py`, `torch/train.py` | forward pass: matmul + bias add; the training step (`zero_grad`/`backward`/`step`) | data load, save/load, MSE |
| tensorflow | `tensorflow/model.py`, `tensorflow/train.py` | forward pass: matmul + bias add; the training step (`GradientTape`/`tape.gradient`/`apply_gradients`) | data load, save/load, MSE |
| onnx | `onnx/export.py`, `onnx/infer.py` | `torch.onnx.export` call, `onnxruntime` session run | loading the trained torch model, sample input |

torch and tensorflow have **two** placeholders each, not one: the forward
pass in `model.py` (already practiced by the time you get here) plus the
training step itself in `train.py`. That second one is deliberate — for
these two frameworks, the autograd/`GradientTape` mechanics are as central
to what's being learned as the forward pass is, unlike numpy/pandas/scipy
where `train.py` really is just boilerplate around a one-shot closed-form
solve.

Note that scipy solves the exact same normal equation as numpy — the point of
this runtime isn't a different algorithm, it's practicing `scipy.linalg`
(LAPACK-backed, lets you hint at matrix structure via `assume_a=`) as an
alternative to `numpy.linalg`.

## Alternative solution: gradient descent

`numpy/model.py`/`numpy/pandas/scipy` all solve OLS in one shot via the
closed-form normal equation. `numpy/model_gd.py` + `numpy/train_gd.py` are an
**alternative solution** to the same problem: fitting via imperative batch
gradient descent instead — computing the loss gradient and updating weights
by hand, in a loop, the way you'd have to for a model with no closed form.
This is deliberately kept as extra files inside the numpy runtime folder
(not a new top-level runtime) since it's the same model, a different
*training method*, not a different library idiom.

`torch/train.py`'s SGD loop is the computation-graph-based counterpart to
this: same iterative gradient-descent idea, but letting autograd compute the
gradients instead of deriving and coding them by hand. Together, `numpy_gd`
(manual gradients) and `torch` (autograd) are the two ends of the "how do I
train something without a closed form" spectrum — worth doing both to feel
the difference.

Run it the same way as any other runtime: `uv run python
models/01-linear-regression/numpy/train_gd.py` (after `model_gd.py`'s
`_step` placeholder is filled in).

## Alternative solutions: framework abstraction levels

torch and tensorflow each have a spectrum from "raw computation graph" up
to "pre-built layer," and it's worth seeing all three rungs of it:

| File | Level | What owns the weights/bias |
|---|---|---|
| `torch/model_raw.py` | lowest | you: plain `torch.Tensor(requires_grad=True)`, no `nn.Module` at all |
| `torch/model.py` | middle | you, via `nn.Parameter`, wrapped in `nn.Module` |
| `torch/model_layer.py` | highest | `nn.Linear` |
| `tensorflow/model.py` | lowest | you, via `tf.Variable`, wrapped in `tf.Module` |
| `tensorflow/model_keras.py` | highest | `tf.keras.layers.Dense` |

`model_raw.py`'s placeholder is in a `_step()` method rather than the
forward pass (already practiced) — it's a manual gradient-descent update
using autograd-computed gradients (`loss.backward()`, then `torch.no_grad()`
+ manual subtraction + manual `.grad.zero_()`), with no `nn.Module` or
`torch.optim` to hide any of it. This is the direct successor to
`numpy/model_gd.py`: numpy_gd computes gradients by hand *and* updates by
hand; `model_raw.py` lets autograd compute the gradients but still updates
by hand; `model.py`/`model_layer.py` hand both jobs to `torch.optim`.

`model_layer.py` and `model_keras.py`'s placeholders are in model
construction, not the forward pass — building an `nn.Linear`/`Dense` layer
is the whole point (it owns its own parameters internally), so there's
nothing left to implement by hand once it exists. Their `train_layer.py`/
`train_keras.py` files carry their own training-step placeholder too, same
as `torch/train.py`/`tensorflow/train.py` — `train_layer.py`'s is the same
`zero_grad`/`backward`/`step` sequence, while `train_keras.py`'s is
`model.compile(...)` + `model.fit(...)` itself, the declarative counterpart
to writing the loop by hand. `train_raw.py` is the one exception: its
training step already lives in `model_raw.py`'s `_step()`, so `train_raw.py`
itself stays fully implemented.

Run them like any other runtime, e.g. `uv run python
models/01-linear-regression/torch/train_raw.py`, `train_layer.py`, or
`models/01-linear-regression/tensorflow/train_keras.py`.

## Running it

```bash
# once, shared by every runtime:
uv run python models/01-linear-regression/data/make_dataset.py

# any runtime, e.g. numpy:
uv run python models/01-linear-regression/numpy/train.py
uv run python models/01-linear-regression/numpy/infer.py

# onnx depends on torch having been trained first:
uv run python models/01-linear-regression/torch/train.py
uv run python models/01-linear-regression/onnx/export.py
uv run python models/01-linear-regression/onnx/infer.py
```

Each runtime's `train.py` prints test-set MSE — since every runtime fits the
same synthetic data with the same true weights `[3.0, -2.0, 0.5]` and bias
`5.0`, once implemented they should all recover parameters close to those
and report similar MSE (bounded below by the injected noise variance).
