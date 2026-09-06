import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import RidgeRegression

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.npz"


def load_model() -> RidgeRegression:
    params = np.load(MODEL_PATH)
    model = RidgeRegression()
    model.weights = params["weights"]
    model.bias = float(params["bias"])
    return model


def main():
    model = load_model()
    sample = np.array([[1.0, -1.0, 0.5, 0.0, 0.0]])
    prediction = model.predict(sample)
    log.info(f"Prediction for {sample.tolist()}: {prediction.tolist()}")


if __name__ == "__main__":
    main()
