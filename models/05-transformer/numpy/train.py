import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from layers import Adam, softmax_cross_entropy
from model import GPT

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "model.npz"

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
    X = np.stack([ids[s : s + block_size] for s in starts]).astype(np.int64)
    Y = np.stack([ids[s + 1 : s + block_size + 1] for s in starts]).astype(np.int64)
    return X, Y


def save_model(model: GPT, path: Path) -> None:
    flat = {k.replace(".", "__"): v for k, v in model.params().items()}
    np.savez(path, vocab_size=model.vocab_size, block_size=model.block_size, **flat)


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]

    tokens = np.load(DATA_DIR / "tokens.npz")
    train_ids, val_ids = tokens["train_ids"], tokens["val_ids"]
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
    optimizer = Adam(model.params(), lr=LR)

    for step in range(N_ITERS):
        X, Y = get_batch(train_ids, BLOCK_SIZE, BATCH_SIZE, rng)
        logits, cache = model.forward(X)
        loss, dlogits = softmax_cross_entropy(logits.reshape(-1, vocab_size), Y.reshape(-1))
        grads = model.backward(dlogits.reshape(BATCH_SIZE, BLOCK_SIZE, vocab_size), cache)
        optimizer.step(grads)

        if step % 200 == 0:
            X_val, Y_val = get_batch(val_ids, BLOCK_SIZE, BATCH_SIZE, rng)
            val_logits, _ = model.forward(X_val)
            val_loss, _ = softmax_cross_entropy(val_logits.reshape(-1, vocab_size), Y_val.reshape(-1))
            log.info(f"step {step}: train loss {loss:.4f}, val loss {val_loss:.4f}")

    save_model(model, MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
