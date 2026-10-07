import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import BERTClassifier

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "model.npz"

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256


def load_model() -> BERTClassifier:
    data = np.load(MODEL_PATH)
    vocab_size, max_len, pad_id = int(data["vocab_size"]), int(data["max_len"]), int(data["pad_id"])
    n_classes = data["classifier__W"].shape[1]
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
    params = model.params()
    for name in params:
        params[name][...] = data[name.replace(".", "__")]
    return model


def encode(message: str, stoi: dict[str, int], max_len: int) -> np.ndarray:
    ids = [stoi["[CLS]"]] + [stoi[ch] for ch in message[: max_len - 1]]
    ids += [stoi["[PAD]"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)

    model = load_model()
    message = "FREE entry! Win a brand new car, text WIN to 80085 now!!!"
    ids = encode(message, vocab["stoi"], model.max_len)[None, :]
    logits, _ = model.forward(ids)
    probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    pred = vocab["label_names"][int(probs.argmax())]
    log.info(f"Message: {message!r}")
    log.info(f"Predicted: {pred}, probs={dict(zip(vocab['label_names'], probs[0].tolist()))}")


if __name__ == "__main__":
    main()
