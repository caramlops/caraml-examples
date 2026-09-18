import logging

import tensorflow as tf

log = logging.getLogger(__name__)


class LogisticRegressionModule(tf.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.weights = tf.Variable(tf.zeros([n_features, n_classes]), name="weights")
        self.bias = tf.Variable(tf.zeros([n_classes]), name="bias")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        log.debug(f"x shape={x.shape}")

        # TODO(you): implement logits = tf.matmul(x, self.weights) +
        # self.bias.
        #   - x has shape (batch, n_features)
        #   - self.weights has shape (n_features, n_classes)
        #   - return raw logits, shape (batch, n_classes) -- do NOT apply
        #     softmax here. train.py's loss
        #     (tf.nn.sparse_softmax_cross_entropy_with_logits) applies
        #     softmax internally in a single, numerically stable fused
        #     op; applying it yourself first would double it.
        raise NotImplementedError("Implement the matmul + add forward pass")
