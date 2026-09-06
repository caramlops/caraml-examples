# 01 · Linear regression

Ordinary least squares: fit `y = X @ weights + bias` to minimize squared
error. The data is synthetic and seeded (`data/make_dataset.py`) so results
are reproducible across runtimes: `y = X @ [3.0, -2.0, 0.5] + 5.0 + noise`.

Every runtime below is fully wired up (data loading, training loop,
evaluation, saving/loading, a sample prediction) except for **one placeholder
per runtime** — the actual mathematical step. Fill it in, then run
`train.py` followed by `infer.py`.

| Runtime | File | Placeholder | Fully implemented |
|---|---|---|---|
| numpy | `numpy/model.py` | OLS normal-equation solve | data load, train loop, save/load, MSE |
| pandas | `pandas/model.py` | OLS solve via centered-mean trick on a DataFrame | CSV load, MSE, save/load |
| scipy | `scipy/model.py` | residual function for `scipy.optimize.least_squares` | fit driver, save/load, MSE |
| torch | `torch/model.py` | forward pass: matmul + bias add | SGD training loop, save/load |
| tensorflow | `tensorflow/model.py` | forward pass: matmul + bias add | `GradientTape` training loop, save/load |
| onnx | `onnx/export.py`, `onnx/infer.py` | `torch.onnx.export` call, `onnxruntime` session run | loading the trained torch model, sample input |

## Running it

```bash
# once, shared by every runtime:
python models/01-linear-regression/data/make_dataset.py

# any runtime, e.g. numpy:
python models/01-linear-regression/numpy/train.py
python models/01-linear-regression/numpy/infer.py

# onnx depends on torch having been trained first:
python models/01-linear-regression/torch/train.py
python models/01-linear-regression/onnx/export.py
python models/01-linear-regression/onnx/infer.py
```

Each runtime's `train.py` prints test-set MSE — since every runtime fits the
same synthetic data with the same true weights `[3.0, -2.0, 0.5]` and bias
`5.0`, once implemented they should all recover parameters close to those
and report similar MSE (bounded below by the injected noise variance).
