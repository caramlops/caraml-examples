import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import LinearRegressionModule

DATA_PATH = Path(__file__).parent.parent / "data" / "linear_regression.npz"
MODEL_PATH = Path(__file__).parent / "model.npz"
EPOCHS = 200
LR = 0.05


def main():
    data = np.load(DATA_PATH)
    X_train = tf.constant(data["X_train"], dtype=tf.float32)
    y_train = tf.constant(data["y_train"], dtype=tf.float32)
    X_test = tf.constant(data["X_test"], dtype=tf.float32)
    y_test = tf.constant(data["y_test"], dtype=tf.float32)

    model = LinearRegressionModule(n_features=X_train.shape[1])
    optimizer = tf.optimizers.SGD(learning_rate=LR)

    for epoch in range(EPOCHS):
        with tf.GradientTape() as tape:
            preds = model(X_train)
            loss = tf.reduce_mean(tf.square(preds - y_train))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 50 == 0:
            print(f"epoch {epoch}: train MSE {loss.numpy():.4f}")

    test_preds = model(X_test)
    test_mse = tf.reduce_mean(tf.square(test_preds - y_test)).numpy()
    print(f"Test MSE: {test_mse:.4f}")

    np.savez(MODEL_PATH, weights=model.weights.numpy(), bias=model.bias.numpy())
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
