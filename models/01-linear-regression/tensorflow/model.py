import tensorflow as tf


class LinearRegressionModule(tf.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.weights = tf.Variable(tf.zeros([n_features, 1]), name="weights")
        self.bias = tf.Variable(tf.zeros([1]), name="bias")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # TODO(you): implement y = x @ weights + bias.
        #   - x has shape (batch, n_features)
        #   - self.weights has shape (n_features, 1)
        #   - use tf.matmul(x, self.weights) to get shape (batch, 1), add
        #     self.bias (broadcasts), then squeeze the trailing dimension so
        #     the output is shape (batch,).
        raise NotImplementedError("Implement the matmul + add forward pass")
