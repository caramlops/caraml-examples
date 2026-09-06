import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import LinearRegressionOLS

MODEL_PATH = Path(__file__).parent / "model.npz"


def load_model() -> LinearRegressionOLS:
    params = np.load(MODEL_PATH)
    model = LinearRegressionOLS()
    model.weights = params["weights"]
    model.bias = float(params["bias"])
    return model


def main():
    model = load_model()
    sample = np.array([[1.0, -1.0, 0.5]])
    prediction = model.predict(sample)
    print(f"Prediction for {sample.tolist()}: {prediction.tolist()}")


if __name__ == "__main__":
    main()
