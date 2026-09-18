import logging

import tensorflow as tf

log = logging.getLogger(__name__)


class RidgeRegressionModule(tf.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.weights = tf.Variable(tf.zeros([n_features, 1]), name="weights")
        self.bias = tf.Variable(tf.zeros([1]), name="bias")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Same matmul + bias-add as plain linear regression -- ridge only
        # changes the loss (see l2_penalty below), not the forward pass.
        log.debug(f"x shape={x.shape}")
        return tf.squeeze(tf.matmul(x, self.weights) + self.bias, axis=-1)

    def l2_penalty(self, alpha: float) -> tf.Tensor:
        return alpha * tf.reduce_sum(self.weights**2)
