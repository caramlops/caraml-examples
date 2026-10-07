import json
import logging
import os
import sys
from pathlib import Path

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


def load_model(vocab_size: int, max_len: int, n_classes: int, pad_id: int) -> BERTClassifier:
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
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.read(str(MODEL_PATH)).expect_partial()
    return model


def encode(message: str, stoi: dict[str, int], max_len: int) -> list[int]:
    ids = [stoi["[CLS]"]] + [stoi[ch] for ch in message[: max_len - 1]]
    ids += [stoi["[PAD]"]] * (max_len - len(ids))
    return ids


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"] + 2
    max_len = vocab["max_len"] + 1

    model = load_model(vocab_size, max_len, len(vocab["label_names"]), vocab["pad_id"])

    message = "FREE entry! Win a brand new car, text WIN to 80085 now!!!"
    ids = tf.constant([encode(message, vocab["stoi"], max_len)])
    logits = model(ids)
    probs = tf.nn.softmax(logits, axis=1)[0]
    pred = vocab["label_names"][int(tf.argmax(probs))]
    log.info(f"Message: {message!r}")
    log.info(f"Predicted: {pred}, probs={dict(zip(vocab['label_names'], probs.numpy().tolist()))}")


if __name__ == "__main__":
    main()
