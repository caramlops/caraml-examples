import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import PoissonRegression

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.npz"


def load_model() -> PoissonRegression:
    params = np.load(MODEL_PATH)
    model = PoissonRegression(n_features=params["weights"].shape[0])
    model.weights = params["weights"]
    model.bias = float(params["bias"])
    return model


def main():
    model = load_model()
    sample = np.array([[1.0, -1.0, 0.5]])
    prediction = model.predict(sample)
    log.info(f"Predicted mean count for {sample.tolist()}: {prediction.tolist()}")


if __name__ == "__main__":
    main()
