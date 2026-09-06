import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import RidgeRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.npz"
N_FEATURES = 5


def main():
    params = np.load(MODEL_PATH)
    model = RidgeRegressionModule(n_features=N_FEATURES)
    model.weights.assign(params["weights"])
    model.bias.assign(params["bias"])

    sample = tf.constant([[1.0, -1.0, 0.5, 0.0, 0.0]], dtype=tf.float32)
    prediction = model(sample)
    log.info(f"Prediction for {sample.numpy().tolist()}: {prediction.numpy().tolist()}")


if __name__ == "__main__":
    main()
