import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import PoissonRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.pt"
N_FEATURES = 3


def load_model() -> PoissonRegressionModule:
    model = PoissonRegressionModule(n_features=N_FEATURES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model


def main():
    model = load_model()
    sample = torch.tensor([[1.0, -1.0, 0.5]])
    with torch.no_grad():
        prediction = torch.exp(model(sample))
    log.info(f"Predicted mean count for {sample.tolist()}: {prediction.tolist()}")


if __name__ == "__main__":
    main()
