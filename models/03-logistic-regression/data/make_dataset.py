"""Generates the synthetic multiclass classification dataset shared by every
runtime.

Three classes, each a Gaussian blob in 4-dimensional feature space, sharing
one covariance matrix. This isn't an incidental simplification: when classes
are Gaussian with *equal* covariance, the Bayes-optimal decision boundary
between any two of them is provably linear, and is exactly of the softmax
(multinomial logistic) form -- so softmax regression is the *correct* model
for this data, not just a convenient one. NOISE_STD is picked to give the
classes deliberate overlap (no seed produces perfectly separable data) so
the confusion matrix and calibration plot in evaluate.py have something
real to show.

Fully implemented on purpose: the point of this model is deriving and
implementing softmax + cross-entropy, not data wrangling.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
SEED = 42
N_SAMPLES_PER_CLASS = 200
N_FEATURES = 4
TEST_FRACTION = 0.2
NOISE_STD = 1.6

# One mean vector per class -- separated enough to be learnable, close
# enough (relative to NOISE_STD) to overlap at the boundaries.
CLASS_MEANS = np.array(
    [
        [2.0, 0.0, 1.0, 0.0],
        [-1.0, 1.7, 0.0, 1.0],
        [-1.0, -1.7, -1.0, -1.0],
    ]
)
N_CLASSES = len(CLASS_MEANS)


def generate() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    X_parts, y_parts = [], []
    for class_idx, mean in enumerate(CLASS_MEANS):
        X_parts.append(rng.normal(loc=mean, scale=NOISE_STD, size=(N_SAMPLES_PER_CLASS, N_FEATURES)))
        y_parts.append(np.full(N_SAMPLES_PER_CLASS, class_idx))
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def split(X: np.ndarray, y: np.ndarray):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * TEST_FRACTION)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main():
    X, y = generate()
    X_train, y_train, X_test, y_test = split(X, y)

    npz_path = DATA_DIR / "logistic_regression.npz"
    np.savez(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=N_CLASSES,
    )

    print(f"Wrote {len(X_train)} train / {len(X_test)} test rows, {N_CLASSES} classes.")
    print(f"  {npz_path}")


if __name__ == "__main__":
    main()
