import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import BERTClassifier

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pt"

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
BATCH_SIZE = 32
N_ITERS = 1500
LR = 3e-3

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(ids: torch.Tensor, labels: torch.Tensor, batch_size: int, device: str):
    idx = torch.randint(0, len(ids), (batch_size,))
    return ids[idx].to(device), labels[idx].to(device)


@torch.no_grad()
def accuracy(model: BERTClassifier, ids: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    logits = model(ids.to(device))
    preds = logits.argmax(dim=1).cpu()
    return (preds == labels).float().mean().item()


def main():
    log.info(f"Using device: {DEVICE}")
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"] + 2
    max_len = vocab["max_len"] + 1
    n_classes = len(vocab["label_names"])
    pad_id = vocab["pad_id"]

    data = np.load(DATA_DIR / "sms.npz")
    train_ids = torch.tensor(data["train_ids"])
    train_labels = torch.tensor(data["train_labels"])
    val_ids = torch.tensor(data["val_ids"])
    val_labels = torch.tensor(data["val_labels"])
    log.debug(f"train_ids shape={train_ids.shape}, n_classes={n_classes}")

    model = BERTClassifier(
        vocab_size=vocab_size,
        max_len=max_len,
        n_classes=n_classes,
        pad_id=pad_id,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(N_ITERS):
        X, Y = get_batch(train_ids, train_labels, BATCH_SIZE, DEVICE)

        # TODO(you): implement one training step using autograd + the
        # optimizer -- same sequence as every other torch runtime.
        #   1. optimizer.zero_grad()
        #   2. logits = model(X) -- shape (batch, n_classes)
        #   3. loss = loss_fn(logits, Y)
        #   4. loss.backward()
        #   5. optimizer.step()
        # Log every 100 steps with validation accuracy:
        #   val_acc = accuracy(model, val_ids, val_labels, DEVICE)
        #   log.info(f"step {step}: train loss {loss.item():.4f}, val accuracy {val_acc:.4f}")
        raise NotImplementedError("Implement one training step")

    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
