import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import PoissonRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "poisson_regression.npz"
MODEL_PATH = Path(__file__).parent / "model.npz"
EPOCHS = 500
LR = 0.05


def main():
    data = np.load(DATA_PATH)
    X_train = tf.constant(data["X_train"], dtype=tf.float32)
    y_train = tf.constant(data["y_train"], dtype=tf.float32)
    X_test = tf.constant(data["X_test"], dtype=tf.float32)
    y_test = tf.constant(data["y_test"], dtype=tf.float32)
    log.debug(f"X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    model = PoissonRegressionModule(n_features=X_train.shape[1])
    optimizer = tf.optimizers.Adam(learning_rate=LR)

    for epoch in range(EPOCHS):
        # TODO(you): implement one training step using GradientTape.
        #   1. `with tf.GradientTape() as tape:` compute
        #      eta = model(X_train) and
        #      loss = tf.reduce_mean(
        #          tf.nn.log_poisson_loss(targets=y_train, log_input=eta))
        #      log_poisson_loss takes the raw linear predictor (log of the
        #      predicted mean) and the actual counts directly, the same
        #      idea as torch's PoissonNLLLoss(log_input=True).
        #   2. grads = tape.gradient(loss, model.trainable_variables)
        #   3. optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Log every 100 epochs:
        #   log.info(f"epoch {epoch}: train loss {loss.numpy():.4f}")
        raise NotImplementedError("Implement one training step")

    test_preds = tf.exp(model(X_test))
    test_mse = tf.reduce_mean(tf.square(test_preds - y_test)).numpy()
    log.info(f"weights={model.weights.numpy()}, bias={model.bias.numpy()}")
    log.info(f"Test MSE: {test_mse:.4f}")

    np.savez(MODEL_PATH, weights=model.weights.numpy(), bias=model.bias.numpy())
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
