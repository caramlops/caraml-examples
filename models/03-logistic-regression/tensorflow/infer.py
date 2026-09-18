import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import LogisticRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.npz"


def load_model() -> LogisticRegressionModule:
    params = np.load(MODEL_PATH)
    n_features, n_classes = params["weights"].shape
    model = LogisticRegressionModule(n_features=n_features, n_classes=n_classes)
    model.weights.assign(params["weights"])
    model.bias.assign(params["bias"])
    return model


def main():
    model = load_model()
    sample = tf.constant([[2.0, 0.0, 1.0, 0.0]], dtype=tf.float32)
    logits = model(sample)
    probs = tf.nn.softmax(logits, axis=1)
    pred = tf.argmax(probs, axis=1)
    log.info(
        f"Prediction for {sample.numpy().tolist()}: class {pred.numpy().tolist()}, "
        f"probs={probs.numpy().tolist()}"
    )


if __name__ == "__main__":
    main()
