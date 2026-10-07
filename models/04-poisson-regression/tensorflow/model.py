import logging

import tensorflow as tf

log = logging.getLogger(__name__)


class PoissonRegressionModule(tf.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.weights = tf.Variable(tf.zeros([n_features, 1]), name="weights")
        self.bias = tf.Variable(tf.zeros([1]), name="bias")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        log.debug(f"x shape={x.shape}")

        # TODO(you): implement eta = tf.matmul(x, self.weights) +
        # self.bias, then tf.squeeze the trailing dimension.
        #   - x has shape (batch, n_features)
        #   - self.weights has shape (n_features, 1)
        #   - return the raw linear predictor eta (= log of the predicted
        #     mean count), shape (batch,) -- do NOT apply exp() here.
        #     train.py's loss (tf.nn.log_poisson_loss) expects log(mean)
        #     directly and applies exp() internally, the same
        #     "pass the pre-link-function value" pattern as
        #     03-logistic-regression's sparse_softmax_cross_entropy_with_logits.
        raise NotImplementedError("Implement the matmul + add forward pass")
