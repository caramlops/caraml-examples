import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import PoissonRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "poisson_regression.npz"
MODEL_PATH = Path(__file__).parent / "model.pt"
EPOCHS = 500
LR = 0.05


def main():
    data = np.load(DATA_PATH)
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)
    log.debug(f"X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    model = PoissonRegressionModule(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.PoissonNLLLoss()  # log_input=True by default

    for epoch in range(EPOCHS):
        # TODO(you): implement one training step using autograd + the
        # optimizer -- same sequence as every other torch runtime.
        #   1. optimizer.zero_grad()
        #   2. eta = model(X_train)
        #   3. loss = loss_fn(eta, y_train) -- PoissonNLLLoss takes the
        #      raw linear predictor (log of the predicted mean, since
        #      log_input=True) and the actual counts directly.
        #   4. loss.backward()
        #   5. optimizer.step()
        # Log every 100 epochs:
        #   log.info(f"epoch {epoch}: train loss {loss.item():.4f}")
        raise NotImplementedError("Implement one training step")

    with torch.no_grad():
        test_preds = torch.exp(model(X_test))
        test_mse = torch.mean((test_preds - y_test) ** 2).item()
    log.info(f"weights={model.weights.detach()}, bias={model.bias.detach()}")
    log.info(f"Test MSE: {test_mse:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
