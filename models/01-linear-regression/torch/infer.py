import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import LinearRegressionModule

MODEL_PATH = Path(__file__).parent / "model.pt"
N_FEATURES = 3


def main():
    model = LinearRegressionModule(n_features=N_FEATURES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    sample = torch.tensor([[1.0, -1.0, 0.5]])
    with torch.no_grad():
        prediction = model(sample)
    print(f"Prediction for {sample.tolist()}: {prediction.tolist()}")


if __name__ == "__main__":
    main()
