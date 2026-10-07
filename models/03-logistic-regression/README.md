# 03 · Logistic regression (multinomial / softmax)

Multiclass classification: given `x`, predict a probability distribution
over `K` classes via `probs = softmax(x @ weights + bias)`, fit by
minimizing the cross-entropy between predicted probabilities and true
labels. The data is synthetic and seeded (`data/make_dataset.py`): three
Gaussian blobs sharing one covariance matrix, in 4-dimensional feature
space, with deliberate overlap.

That dataset design isn't an arbitrary simplification: when classes are
Gaussian with *equal* covariance, the Bayes-optimal decision boundary
between any two of them is provably linear, and is exactly of the softmax
form — so softmax regression is the mathematically correct model for this
data, not just a convenient linear approximation. A good fit should land
around 75% test accuracy; the rest is genuine, irreducible overlap between
the blobs, not a bug.

**Binary logistic regression** is the `K = 2` special case of everything
here — softmax over two classes reduces algebraically to the sigmoid
function you may already know (`softmax([z, 0])_0 = 1 / (1 + exp(-z)) =
sigmoid(z)`), and cross-entropy over two classes reduces to the familiar
binary cross-entropy. There's no separate scaffold for it: the same
`LogisticRegression`/`LogisticRegressionModule` classes below handle any
`K`, including 2, without modification.

**Why this model has no closed form, unlike `01`/`02`**: OLS and ridge have
a normal equation because their loss is quadratic in the parameters — set
the gradient to zero, solve a linear system, done in one step. Cross-entropy
loss under a softmax is not quadratic; setting its gradient to zero gives a
system of transcendental equations with no algebraic solution. Every
runtime below is therefore doing *some* form of iterative optimization —
there's no "closed-form vs. gradient-descent" contrast to draw the way
`01`/`02`'s `numpy_gd` alternative did, because gradient-based fitting isn't
an alternative here, it's the only option.

| Runtime | File(s) | Placeholder(s) | Fully implemented |
|---|---|---|---|
| numpy | `numpy/model.py` | numerically stable softmax; cross-entropy gradient step | data load, train loop, save/load, accuracy |
| scipy | `scipy/model.py` | combined cross-entropy loss + analytic gradient for `scipy.optimize.minimize` | fit driver, save/load, accuracy |
| torch | `torch/model.py`, `torch/train.py` | forward pass (raw logits, no softmax); the training step (`zero_grad`/`backward`/`step`) | data load, save/load, accuracy |
| tensorflow | `tensorflow/model.py`, `tensorflow/train.py` | forward pass (raw logits); the training step (`GradientTape`/`tape.gradient`/`apply_gradients`) | data load, save/load, accuracy |
| onnx | `onnx/export.py`, `onnx/infer.py` | `torch.onnx.export` call, `onnxruntime` session run | loading the trained torch model, sample input |

**No pandas runtime here, unlike `01`/`02`.** For OLS/ridge, pandas earned
its place by practicing a genuinely different *derivation* — mean-centering
instead of augmenting `X` with a ones-column. Logistic regression has no
closed form for pandas to derive an alternative to: a pandas runtime here
would just be the exact same gradient-descent algorithm as `numpy/model.py`,
with `.values` pulled out of a DataFrame at the boundary and no new
technique to practice in between. That's not a distinct runtime, it's the
same file with extra ceremony, so it's dropped.

## The one gotcha specific to this model: don't apply softmax twice

`torch/model.py` and `tensorflow/model.py`'s forward pass returns **raw
logits**, not probabilities — deliberately. `nn.CrossEntropyLoss` (torch)
and `tf.nn.sparse_softmax_cross_entropy_with_logits` (tensorflow) both take
raw logits and integer class labels directly, and apply softmax internally
as a single, numerically fused operation combined with the loss. If you
apply softmax yourself in the forward pass *and* hand the result to one of
these loss functions, nothing errors — you just silently apply softmax
twice, and training quietly breaks. `numpy`/`scipy` don't have this trap,
since they implement softmax and cross-entropy separately by hand rather
than through a fused loss function.

## `evaluate.py`: confusion matrix + calibration plot

`uv run python models/03-logistic-regression/evaluate.py` (after
`numpy/train.py` has been run) loads `numpy/model.npz`, evaluates it on the
test split, and writes two plots to `plots/`:

- **`confusion_matrix.png`** — true class (rows) vs. predicted class
  (columns), counts.
- **`calibration.png`** — a reliability diagram: bins predictions by their
  top-1 confidence and plots each bin's mean confidence against its actual
  accuracy. Points on the diagonal are well-calibrated; points below it mean
  the model is overconfident in that range, points above mean it's
  underconfident.

This always evaluates the numpy runtime specifically — every runtime fits
the same data to the same objective, so there's nothing to learn by
re-plotting the same two charts six times. It's fully implemented (not a
placeholder): it's analysis/reporting on the model you already built, not a
new derivation.

## Running it

```bash
# once, shared by every runtime:
uv run python models/03-logistic-regression/data/make_dataset.py

# any runtime, e.g. numpy:
uv run python models/03-logistic-regression/numpy/train.py
uv run python models/03-logistic-regression/numpy/infer.py

# confusion matrix + calibration plot (needs numpy/train.py run first):
uv run python models/03-logistic-regression/evaluate.py

# onnx depends on torch having been trained first:
uv run python models/03-logistic-regression/torch/train.py
uv run python models/03-logistic-regression/onnx/export.py
uv run python models/03-logistic-regression/onnx/infer.py
```

Each runtime's `train.py` prints test-set accuracy — since every runtime
fits the same synthetic data to the same cross-entropy objective, once
implemented they should all land close to the same accuracy (~75%, bounded
above by the genuine class overlap baked into the dataset).
