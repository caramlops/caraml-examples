import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import RidgeRegressionLSQ

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.npz"
N_FEATURES = 5


def main():
    data = np.load(MODEL_PATH)
    model = RidgeRegressionLSQ(n_features=N_FEATURES)
    model.params = data["params"]

    sample = np.array([[1.0, -1.0, 0.5, 0.0, 0.0]])
    prediction = model.predict(sample)
    log.info(f"Prediction for {sample.tolist()}: {prediction.tolist()}")


if __name__ == "__main__":
    main()
