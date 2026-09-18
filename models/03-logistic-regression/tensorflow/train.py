import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import LogisticRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "logistic_regression.npz"
MODEL_PATH = Path(__file__).parent / "model.npz"
EPOCHS = 500
LR = 0.5


def main():
    data = np.load(DATA_PATH)
    X_train = tf.constant(data["X_train"], dtype=tf.float32)
    y_train = tf.constant(data["y_train"], dtype=tf.int32)
    X_test = tf.constant(data["X_test"], dtype=tf.float32)
    y_test = tf.constant(data["y_test"], dtype=tf.int32)
    n_classes = int(data["n_classes"])
    log.debug(f"X_train shape={X_train.shape}, n_classes={n_classes}")

    model = LogisticRegressionModule(n_features=X_train.shape[1], n_classes=n_classes)
    optimizer = tf.optimizers.SGD(learning_rate=LR)

    for epoch in range(EPOCHS):
        # TODO(you): implement one training step using GradientTape.
        #   1. `with tf.GradientTape() as tape:` compute
        #      logits = model(X_train) and
        #      loss = tf.reduce_mean(
        #          tf.nn.sparse_softmax_cross_entropy_with_logits(
        #              labels=y_train, logits=logits))
        #      sparse_softmax_cross_entropy_with_logits takes raw logits
        #      and integer class labels directly (no one-hot encoding, no
        #      manual softmax) -- it fuses softmax + cross-entropy into
        #      one numerically stable op, the same idea as torch's
        #      CrossEntropyLoss.
        #   2. grads = tape.gradient(loss, model.trainable_variables)
        #   3. optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Log every 100 epochs:
        #   log.info(f"epoch {epoch}: train loss {loss.numpy():.4f}")
        raise NotImplementedError("Implement one training step")

    test_preds = tf.argmax(model(X_test), axis=1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(test_preds == y_test, tf.float32)).numpy()
    log.info(f"weights={model.weights.numpy()}, bias={model.bias.numpy()}")
    log.info(f"Test accuracy: {accuracy:.4f}")

    np.savez(MODEL_PATH, weights=model.weights.numpy(), bias=model.bias.numpy())
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
