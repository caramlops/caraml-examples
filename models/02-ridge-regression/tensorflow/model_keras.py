import logging

import tensorflow as tf

log = logging.getLogger(__name__)


def build_model(n_features: int, alpha: float) -> tf.keras.Model:
    """The Keras counterpart to model.py's raw tf.Module -- but ridge's L2
    penalty is declared on the layer itself instead of added to the loss
    by hand."""
    log.debug(f"n_features={n_features}, alpha={alpha}")

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer="zeros",
                bias_initializer="zeros",
                kernel_regularizer=tf.keras.regularizers.l2(alpha),
            ),
        ]
    )
