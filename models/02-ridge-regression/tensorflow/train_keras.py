from model_keras import build_model
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "ridge_regression.npz"
MODEL_PATH = Path(__file__).parent / "model_keras.keras"
EPOCHS = 200
LR = 0.05
ALPHA = 5.0


def main():
    data = np.load(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    log.debug(f"X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    model = build_model(n_features=X_train.shape[1], alpha=ALPHA)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LR), loss="mse")

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=X_train.shape[0], verbose=0)

    for epoch in range(0, EPOCHS, 50):
        loss = history.history["loss"][epoch]
        log.info(f"epoch {epoch}: train loss (MSE + penalty) {loss:.4f}")

    # model.evaluate() would report the compiled loss *plus* the Dense
    # layer's regularization penalty (Keras folds model.losses into every
    # forward pass, train or not) -- compute plain MSE by hand instead, so
    # this number means the same thing as every other runtime's Test MSE.
    test_preds = model.predict(X_test, verbose=0).squeeze(-1)
    test_mse = np.mean((test_preds - y_test) ** 2)
    weights, bias = model.get_weights()
    log.info(f"weights={weights}, bias={bias}")
    log.info(f"Test MSE: {test_mse:.4f}")

    model.save(MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
