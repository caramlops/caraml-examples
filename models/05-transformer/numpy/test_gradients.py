"""Finite-difference gradient check for attention.py's hand-derived
backward pass.

Fully implemented -- this is a correctness *tool*, not a modeling
placeholder. Run it after implementing attention.py's forward/backward:

    uv run python models/05-transformer/numpy/test_gradients.py

For every parameter in a tiny (deliberately small, so this runs in
seconds) GPT instance, it numerically estimates d(loss)/d(param) by
perturbing each element and re-running the forward pass (the definition of
a derivative, computed directly rather than analytically), and compares
that against your backward()'s analytic gradient. A correct
backward pass should get relative error well under 1e-4; anything larger
means there's a bug in the derivation or the code, specifically in
attention.py (every other layer here is already correct and tested this
same way).
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from layers import softmax_cross_entropy
from model import GPT

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
    vocab_size, block_size, batch_size = 6, 4, 2
    model = GPT(
        vocab_size=vocab_size, block_size=block_size, d_model=4, n_heads=2, n_layers=1, d_ff=8, seed=0
    )

    ids = rng.integers(0, vocab_size, size=(batch_size, block_size))
    targets = rng.integers(0, vocab_size, size=(batch_size, block_size))

    def loss_fn() -> float:
        logits, _ = model.forward(ids)
        loss, _ = softmax_cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        return loss

    logits, cache = model.forward(ids)
    loss, dlogits = softmax_cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    analytic_grads = model.backward(dlogits.reshape(batch_size, block_size, vocab_size), cache)
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
