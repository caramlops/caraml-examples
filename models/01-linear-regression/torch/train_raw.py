import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_raw import LinearRegressionRaw  # noqa

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "linear_regression.npz"
MODEL_PATH = Path(__file__).parent / "model_raw.npz"


def main():
    data = np.load(DATA_PATH)
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)
    log.debug(f"X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    model = LinearRegressionRaw(n_features=X_train.shape[1])
    model.fit(X_train, y_train)

    with torch.no_grad():
        test_mse = torch.mean((model.predict(X_test) - y_test) ** 2).item()
    log.info(f"weights={model.weights.detach()}, bias={model.bias.detach()}")
    log.info(f"Test MSE: {test_mse:.4f}")

    np.savez(
        MODEL_PATH,
        weights=model.weights.detach().numpy(),
        bias=model.bias.detach().numpy(),
    )
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
