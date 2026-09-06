import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import LinearRegressionModule

MODEL_PATH = Path(__file__).parent / "model.npz"
N_FEATURES = 3


def main():
    params = np.load(MODEL_PATH)
    model = LinearRegressionModule(n_features=N_FEATURES)
    model.weights.assign(params["weights"])
    model.bias.assign(params["bias"])

    sample = tf.constant([[1.0, -1.0, 0.5]], dtype=tf.float32)
    prediction = model(sample)
    print(f"Prediction for {sample.numpy().tolist()}: {prediction.numpy().tolist()}")


if __name__ == "__main__":
    main()
