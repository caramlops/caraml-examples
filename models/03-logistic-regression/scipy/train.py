import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import LogisticRegression

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "logistic_regression.npz"
MODEL_PATH = Path(__file__).parent / "model.npz"


def main():
    data = np.load(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    n_classes = int(data["n_classes"])
    log.debug(f"X_train shape={X_train.shape}, n_classes={n_classes}")

    model = LogisticRegression(n_features=X_train.shape[1], n_classes=n_classes)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    log.info(f"weights={model.weights}, bias={model.bias}")
    log.info(f"Test accuracy: {accuracy:.4f}")

    np.savez(MODEL_PATH, weights=model.weights, bias=model.bias)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
