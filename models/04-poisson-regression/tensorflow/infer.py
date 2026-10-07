import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import PoissonRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.npz"


def load_model() -> PoissonRegressionModule:
    params = np.load(MODEL_PATH)
    n_features = params["weights"].shape[0]
    model = PoissonRegressionModule(n_features=n_features)
    model.weights.assign(params["weights"])
    model.bias.assign(params["bias"])
    return model


def main():
    model = load_model()
    sample = tf.constant([[1.0, -1.0, 0.5]], dtype=tf.float32)
    prediction = tf.exp(model(sample))
    log.info(f"Predicted mean count for {sample.numpy().tolist()}: {prediction.numpy().tolist()}")


if __name__ == "__main__":
    main()
