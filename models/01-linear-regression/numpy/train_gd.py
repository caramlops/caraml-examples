import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_gd import LinearRegressionGD

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "linear_regression.npz"
MODEL_PATH = Path(__file__).parent / "model_gd.npz"


def main():
    data = np.load(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    log.debug(f"X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    model = LinearRegressionGD(n_features=X_train.shape[1])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = np.mean((preds - y_test) ** 2)
    log.info(f"weights={model.weights}, bias={model.bias}")
    log.info(f"Test MSE: {mse:.4f}")

    np.savez(MODEL_PATH, weights=model.weights, bias=model.bias)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
