"""Generates the synthetic linear-regression dataset shared by every runtime.

y = X @ TRUE_WEIGHTS + TRUE_BIAS + noise

Fully implemented on purpose: the point of this model is practicing the OLS
solve / gradient step, not data wrangling.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
SEED = 42
N_SAMPLES = 500
TRUE_WEIGHTS = np.array([3.0, -2.0, 0.5])
TRUE_BIAS = 5.0
NOISE_STD = 1.0
TEST_FRACTION = 0.2


def generate() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n_features = len(TRUE_WEIGHTS)
    X = rng.normal(size=(N_SAMPLES, n_features))
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

    npz_path = DATA_DIR / "linear_regression.npz"
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
    print(f"  {npz_path}")
    print(f"  {train_csv}")
    print(f"  {test_csv}")


if __name__ == "__main__":
    main()
