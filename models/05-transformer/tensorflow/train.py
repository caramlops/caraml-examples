import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "checkpoint" / "model"

BLOCK_SIZE = 64
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
BATCH_SIZE = 32
N_ITERS = 3000
LR = 3e-3


def get_batch(ids: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator):
    starts = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    X = np.stack([ids[s : s + block_size] for s in starts]).astype(np.int32)
    Y = np.stack([ids[s + 1 : s + block_size + 1] for s in starts]).astype(np.int32)
    return tf.constant(X), tf.constant(Y)


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]

    tokens = np.load(DATA_DIR / "tokens.npz")
    train_ids = tokens["train_ids"]
    log.debug(f"train_ids shape={train_ids.shape}, vocab_size={vocab_size}")

    rng = np.random.default_rng(42)
    model = GPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    )
    optimizer = tf.optimizers.AdamW(learning_rate=LR)

    for step in range(N_ITERS):
        X, Y = get_batch(train_ids, BLOCK_SIZE, BATCH_SIZE, rng)

        # TODO(you): implement one training step using GradientTape.
        #   1. `with tf.GradientTape() as tape:` compute
        #      logits = model(X) -- shape (batch, block_size, vocab_size)
        #      -- and
        #      loss = tf.reduce_mean(
        #          tf.nn.sparse_softmax_cross_entropy_with_logits(
        #              labels=Y, logits=logits))
        #   2. grads = tape.gradient(loss, model.trainable_variables)
        #   3. optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Log every 200 steps:
        #   log.info(f"step {step}: train loss {loss.numpy():.4f}")
        raise NotImplementedError("Implement one training step")

    # GPT is a plain tf.Module, not a tf.keras.Model, so it has no
    # save_weights() of its own -- tf.train.Checkpoint is the generic way
    # to save/restore any tf.Module's variables.
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.write(str(MODEL_PATH))
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
