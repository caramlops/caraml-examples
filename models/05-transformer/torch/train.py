import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pt"

BLOCK_SIZE = 64
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
BATCH_SIZE = 32
N_ITERS = 3000
LR = 3e-3

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(
    ids: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,))
    X = torch.stack([ids[s : s + block_size] for s in starts]).to(device)
    Y = torch.stack([ids[s + 1 : s + block_size + 1] for s in starts]).to(device)
    return X, Y


def main():
    log.info(f"Using device: {DEVICE}")
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]

    tokens = np.load(DATA_DIR / "tokens.npz")
    train_ids = torch.tensor(tokens["train_ids"].astype(np.int64))
    val_ids = torch.tensor(tokens["val_ids"].astype(np.int64))
    log.debug(f"train_ids shape={train_ids.shape}, vocab_size={vocab_size}")

    model = GPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(N_ITERS):
        X, Y = get_batch(train_ids, BLOCK_SIZE, BATCH_SIZE, DEVICE)

        # TODO(you): implement one training step using autograd + the
        # optimizer -- same sequence as every other torch runtime.
        #   1. optimizer.zero_grad()
        #   2. logits = model(X) -- shape (batch, block_size, vocab_size)
        #   3. loss = loss_fn(logits.view(-1, vocab_size), Y.view(-1)) --
        #      CrossEntropyLoss wants (N, vocab_size) logits and (N,)
        #      integer targets, so flatten the batch and sequence
        #      dimensions together first.
        #   4. loss.backward()
        #   5. optimizer.step()
        # Log every 200 steps:
        #   log.info(f"step {step}: train loss {loss.item():.4f}")
        raise NotImplementedError("Implement one training step")

    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
