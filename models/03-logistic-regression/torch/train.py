import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import LogisticRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "logistic_regression.npz"
MODEL_PATH = Path(__file__).parent / "model.pt"
EPOCHS = 500
LR = 0.5


def main():
    data = np.load(DATA_PATH)
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)
    n_classes = int(data["n_classes"])
    log.debug(f"X_train shape={X_train.shape}, n_classes={n_classes}")

    model = LogisticRegressionModule(n_features=X_train.shape[1], n_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        # TODO(you): implement one training step using autograd + the
        # optimizer -- same sequence as the linear-regression torch
        # runtime.
        #   1. optimizer.zero_grad()
        #   2. logits = model(X_train)
        #   3. loss = loss_fn(logits, y_train) -- CrossEntropyLoss takes
        #      raw logits (shape (batch, n_classes)) and integer class
        #      labels (shape (batch,)) directly, not one-hot vectors or
        #      pre-softmaxed probabilities.
        #   4. loss.backward()
        #   5. optimizer.step()
        # Log every 100 epochs:
        #   log.info(f"epoch {epoch}: train loss {loss.item():.4f}")
        raise NotImplementedError("Implement one training step")

    with torch.no_grad():
        test_preds = model(X_test).argmax(dim=1)
        accuracy = (test_preds == y_test).float().mean().item()
    log.info(f"weights={model.weights.detach()}, bias={model.bias.detach()}")
    log.info(f"Test accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
