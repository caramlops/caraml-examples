import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import LinearRegressionLSQ

MODEL_PATH = Path(__file__).parent / "model.npz"
N_FEATURES = 3


def main():
    data = np.load(MODEL_PATH)
    model = LinearRegressionLSQ(n_features=N_FEATURES)
    model.params = data["params"]

    sample = np.array([[1.0, -1.0, 0.5]])
    prediction = model.predict(sample)
    print(f"Prediction for {sample.tolist()}: {prediction.tolist()}")


if __name__ == "__main__":
    main()
