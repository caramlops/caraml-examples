"""Generates a synthetic dataset with deliberate multicollinearity.

Three independent latent features (z0, z1, z2) drive y. Two extra columns
(x3, x4) are near-linear combinations of the others with only a hair of
noise added -- i.e. columns that are almost redundant given the rest of X.
Their true weight is 0 (they carry no real signal), but because they're
highly correlated with columns that DO matter, plain OLS's normal equation
involves inverting a near-singular X^T X: small changes in the sample can
swing the fitted weights on the collinear columns wildly, even though
predictions stay reasonable. Ridge's added `alpha * I` term keeps that
matrix comfortably invertible and pulls those spurious weights toward zero.

Fully implemented on purpose: the point of this model is the ridge penalty,
not data wrangling.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
SEED = 42
N_SAMPLES = 200
TRUE_WEIGHTS = np.array([2.0, -1.5, 1.0, 0.0, 0.0])
TRUE_BIAS = 4.0
NOISE_STD = 1.0
COLLINEAR_NOISE_STD = 0.05
TEST_FRACTION = 0.2


def generate() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    z0, z1, z2 = rng.normal(size=(3, N_SAMPLES))

    x3 = z0 + z1 + rng.normal(scale=COLLINEAR_NOISE_STD, size=N_SAMPLES)
    x4 = 0.5 * z0 - 0.5 * z2 + rng.normal(scale=COLLINEAR_NOISE_STD, size=N_SAMPLES)

    X = np.column_stack([z0, z1, z2, x3, x4])
    noise = rng.normal(scale=NOISE_STD, size=N_SAMPLES)
    y = X @ TRUE_WEIGHTS + TRUE_BIAS + noise
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

    npz_path = DATA_DIR / "ridge_regression.npz"
    np.savez(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    columns = [f"x{i}" for i in range(X.shape[1])]
    train_df = pd.DataFrame(X_train, columns=columns)
    train_df["y"] = y_train
    test_df = pd.DataFrame(X_test, columns=columns)
    test_df["y"] = y_test

    train_csv, test_csv = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Wrote {len(X_train)} train / {len(X_test)} test rows.")
    print(f"  condition number of X_train^T X_train: {np.linalg.cond(X_train.T @ X_train):.2e}")
    print(f"  {npz_path}")
    print(f"  {train_csv}")
    print(f"  {test_csv}")


if __name__ == "__main__":
    main()
