from model_layer import LinearRegressionLayer
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "linear_regression.npz"
MODEL_PATH = Path(__file__).parent / "model_layer.pt"
EPOCHS = 200
LR = 0.05


def main():
    data = np.load(DATA_PATH)
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)
    log.debug(f"X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    model = LinearRegressionLayer(n_features=X_train.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        preds = model(X_train)

        loss = loss_fn(preds, y_train)
        loss.backward()

        optimizer.step()

        if epoch % 50 == 0:
            log.info(f"epoch {epoch}: train MSE {loss.item():.4f}")

    with torch.no_grad():
        test_mse = loss_fn(model(X_test), y_test).item()
    log.info(f"weights={model.linear.weight.detach()}, bias={
             model.linear.bias.detach()}")
    log.info(f"Test MSE: {test_mse:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
