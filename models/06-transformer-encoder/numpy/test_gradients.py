"""Finite-difference gradient check for attention.py's hand-derived
backward pass -- same tool, same idea as 05-transformer's, adapted for
this model's padding-masked bidirectional attention and classification
head.

    uv run python models/06-transformer-encoder/numpy/test_gradients.py

A correct implementation gets relative error well under 1e-4 on every
parameter; a real bug shows up as a specific named parameter failing
(almost always one of attn.Wq/Wk/Wv/Wo, since every other layer here is
already correct and was verified this same way before being shipped).
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from layers import softmax_cross_entropy
from model import BERTClassifier

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)


def numerical_gradient(loss_fn, param: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        loss_plus = loss_fn()
        param[idx] = orig - eps
        loss_minus = loss_fn()
        param[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)))


def main():
    rng = np.random.default_rng(0)
    char_vocab_size, max_len, n_classes, batch_size = 4, 5, 2, 2
    cls_id, pad_id = char_vocab_size, char_vocab_size + 1
    vocab_size = char_vocab_size + 2

    model = BERTClassifier(
        vocab_size=vocab_size,
        max_len=max_len,
        n_classes=n_classes,
        pad_id=pad_id,
        d_model=4,
        n_heads=2,
        n_layers=1,
        d_ff=8,
        seed=0,
    )

    # Build ids with [CLS] at position 0 and some trailing [PAD] -- the
    # exact shape the real data has, so the mask actually gets exercised.
    ids = np.full((batch_size, max_len), pad_id, dtype=np.int64)
    ids[:, 0] = cls_id
    ids[0, 1:4] = rng.integers(0, char_vocab_size, size=3)
    ids[1, 1:3] = rng.integers(0, char_vocab_size, size=2)
    labels = rng.integers(0, n_classes, size=batch_size)

    def loss_fn() -> float:
        logits, _ = model.forward(ids)
        loss, _ = softmax_cross_entropy(logits, labels)
        return loss

    logits, cache = model.forward(ids)
    loss, dlogits = softmax_cross_entropy(logits, labels)
    analytic_grads = model.backward(dlogits, cache)
    log.info(f"Initial loss: {loss:.4f}")

    params = model.params()
    all_ok = True
    for name, param in params.items():
        numeric = numerical_gradient(loss_fn, param)
        err = relative_error(numeric, analytic_grads[name])
        status = "OK" if err < 1e-4 else "MISMATCH"
        if err >= 1e-4:
            all_ok = False
        log.info(f"{name:20s} relative error={err:.2e}  [{status}]")

    if all_ok:
        log.info("All gradients match finite differences. Attention backward is correct.")
    else:
        log.info("Some gradients don't match -- check attention.py's backward derivation.")


if __name__ == "__main__":
    main()
