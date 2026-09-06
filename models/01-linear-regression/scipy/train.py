import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import LinearRegressionLSQ

DATA_PATH = Path(__file__).parent.parent / "data" / "linear_regression.npz"
MODEL_PATH = Path(__file__).parent / "model.npz"


def main():
    data = np.load(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    model = LinearRegressionLSQ(n_features=X_train.shape[1])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = np.mean((preds - y_test) ** 2)
    print(f"params={model.params}")
    print(f"Test MSE: {mse:.4f}")

    np.savez(MODEL_PATH, params=model.params)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
