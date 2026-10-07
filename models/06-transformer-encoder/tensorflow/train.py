import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from model import BERTClassifier

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "checkpoint" / "model"

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
BATCH_SIZE = 32
N_ITERS = 1500
LR = 3e-3


def get_batch(ids: np.ndarray, labels: np.ndarray, batch_size: int, rng: np.random.Generator):
    idx = rng.integers(0, len(ids), size=batch_size)
    return tf.constant(ids[idx]), tf.constant(labels[idx])


def accuracy(model: BERTClassifier, ids: np.ndarray, labels: np.ndarray) -> float:
    logits = model(tf.constant(ids))
    preds = tf.argmax(logits, axis=1, output_type=tf.int64).numpy()
    return float((preds == labels).mean())


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"] + 2
    max_len = vocab["max_len"] + 1
    n_classes = len(vocab["label_names"])
    pad_id = vocab["pad_id"]

    data = np.load(DATA_DIR / "sms.npz")
    train_ids, train_labels = data["train_ids"], data["train_labels"]
    val_ids, val_labels = data["val_ids"], data["val_labels"]
    log.debug(f"train_ids shape={train_ids.shape}, n_classes={n_classes}")

    rng = np.random.default_rng(42)
    model = BERTClassifier(
        vocab_size=vocab_size,
        max_len=max_len,
        n_classes=n_classes,
        pad_id=pad_id,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    )
    optimizer = tf.optimizers.AdamW(learning_rate=LR)

    for step in range(N_ITERS):
        X, Y = get_batch(train_ids, train_labels, BATCH_SIZE, rng)

        # TODO(you): implement one training step using GradientTape.
        #   1. `with tf.GradientTape() as tape:` compute
        #      logits = model(X) and
        #      loss = tf.reduce_mean(
        #          tf.nn.sparse_softmax_cross_entropy_with_logits(
        #              labels=Y, logits=logits))
        #   2. grads = tape.gradient(loss, model.trainable_variables)
        #   3. optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Log every 100 steps with validation accuracy:
        #   val_acc = accuracy(model, val_ids, val_labels)
        #   log.info(f"step {step}: train loss {loss.numpy():.4f}, val accuracy {val_acc:.4f}")
        raise NotImplementedError("Implement one training step")

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.write(str(MODEL_PATH))
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
