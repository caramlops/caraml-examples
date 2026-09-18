import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import LogisticRegression

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.json"
FEATURE_COLUMNS = ["x0", "x1", "x2", "x3"]


def load_model() -> LogisticRegression:
    with open(MODEL_PATH) as f:
        params = json.load(f)
    weights = np.array(params["weights"])
    n_classes = weights.shape[1]
    model = LogisticRegression(FEATURE_COLUMNS, n_classes=n_classes)
    model.weights = pd.DataFrame(weights, index=FEATURE_COLUMNS)
    model.bias = pd.Series(params["bias"])
    return model


def main():
    model = load_model()
    sample = pd.DataFrame([[2.0, 0.0, 1.0, 0.0]], columns=FEATURE_COLUMNS)
    probs = model.predict_proba(sample)
    pred = probs.idxmax(axis=1)
    log.info(f"Prediction for {sample.values.tolist()}: class {pred.tolist()}, probs=\n{probs}")


if __name__ == "__main__":
    main()
