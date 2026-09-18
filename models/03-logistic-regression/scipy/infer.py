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

MODEL_PATH = Path(__file__).parent / "model.npz"


def load_model() -> LogisticRegression:
    params = np.load(MODEL_PATH)
    n_features, n_classes = params["weights"].shape
    model = LogisticRegression(n_features=n_features, n_classes=n_classes)
    model.weights = params["weights"]
    model.bias = params["bias"]
    return model


def main():
    model = load_model()
    sample = np.array([[2.0, 0.0, 1.0, 0.0]])
    probs = model.predict_proba(sample)
    pred = probs.argmax(axis=1)
    log.info(f"Prediction for {sample.tolist()}: class {pred.tolist()}, probs={probs.tolist()}")


if __name__ == "__main__":
    main()
