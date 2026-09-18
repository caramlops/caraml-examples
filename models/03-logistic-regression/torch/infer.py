import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import LogisticRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.pt"
N_FEATURES = 4
N_CLASSES = 3


def load_model() -> LogisticRegressionModule:
    model = LogisticRegressionModule(n_features=N_FEATURES, n_classes=N_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model


def main():
    model = load_model()
    sample = torch.tensor([[2.0, 0.0, 1.0, 0.0]])
    with torch.no_grad():
        logits = model(sample)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
    log.info(f"Prediction for {sample.tolist()}: class {pred.tolist()}, probs={probs.tolist()}")


if __name__ == "__main__":
    main()
