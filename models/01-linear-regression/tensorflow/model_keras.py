import logging

import tensorflow as tf

log = logging.getLogger(__name__)


def build_model(n_features: int) -> tf.keras.Model:
    """The Keras counterpart to model.py's raw tf.Module -- same
    y = x @ weights + bias, but built from a Dense layer instead of
    hand-declared tf.Variables."""
    log.debug(f"n_features={n_features}")

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(units=1, kernel_initializer="zeros", bias_initializer="zeros"),
        ]
    )
