"""Generates the synthetic Poisson-regression (count data) dataset shared by
every runtime.

y_i ~ Poisson(mu_i), mu_i = exp(X_i @ TRUE_WEIGHTS + TRUE_BIAS)

Counts, not continuous values: y is a non-negative integer, and its own
variance grows with its mean (Var(Y) = E[Y] = mu for a true Poisson
variable) -- plain OLS's constant-variance assumption doesn't hold here,
which is exactly why this model needs its own fitting algorithm (IRLS)
instead of reusing 01-linear-regression's normal equation.

Fully implemented on purpose: the point of this model is deriving and
implementing IRLS, not data wrangling.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
SEED = 42
N_SAMPLES = 600
TRUE_WEIGHTS = np.array([0.4, 0.3, -0.5])
TRUE_BIAS = 1.0
TEST_FRACTION = 0.2


def generate() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n_features = len(TRUE_WEIGHTS)
    X = rng.normal(scale=0.6, size=(N_SAMPLES, n_features))
    mu = np.exp(X @ TRUE_WEIGHTS + TRUE_BIAS)
    y = rng.poisson(mu)
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

    npz_path = DATA_DIR / "poisson_regression.npz"
    np.savez(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print(f"Wrote {len(X_train)} train / {len(X_test)} test rows.")
    print(f"  y range: [{y.min()}, {y.max()}], mean={y.mean():.2f}")
    print(f"  {npz_path}")


if __name__ == "__main__":
    main()
