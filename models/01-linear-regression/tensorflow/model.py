import logging

import tensorflow as tf

log = logging.getLogger(__name__)


class LinearRegressionModule(tf.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.weights = tf.Variable(tf.zeros([n_features, 1]), name="weights")
        self.bias = tf.Variable(tf.zeros([1]), name="bias")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        log.debug(f"x shape={x.shape}")

        result = tf.matmul(x, self.weights) + self.bias

        return tf.squeeze(result, axis=-1)
