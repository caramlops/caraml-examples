"""Downloads the tiny-Shakespeare corpus and tokenizes it at the character
level -- the same dataset (and the same tokenization scheme) Karpathy's
char-rnn/nanoGPT tutorials use, chosen because it's tiny (~1MB), freely
available, and small enough to train a real-but-small transformer on
locally without needing a datacenter.

Per CLAUDE.md's data convention ("prefer synthetic... unless a real dataset
materially matters"): a from-scratch language model trained on synthetic
gibberish text wouldn't teach you anything about whether your
implementation actually learns language structure -- the whole point only
works with real, structured text.

Fully implemented on purpose: the point of this model is deriving and
implementing attention, not data wrangling.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CORPUS_PATH = DATA_DIR / "input.txt"
SEED = 42
VAL_FRACTION = 0.1


def download_corpus() -> str:
    if not CORPUS_PATH.exists():
        print(f"Downloading corpus from {CORPUS_URL} ...")
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
    return CORPUS_PATH.read_text()


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def main():
    text = download_corpus()
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)

    ids = np.array([stoi[ch] for ch in text], dtype=np.uint16)
    n_val = int(len(ids) * VAL_FRACTION)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    tokens_path = DATA_DIR / "tokens.npz"
    np.savez(tokens_path, train_ids=train_ids, val_ids=val_ids)

    vocab_path = DATA_DIR / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, f)

    print(f"Corpus: {len(text)} characters, vocab size {vocab_size}.")
    print(f"Wrote {len(train_ids)} train / {len(val_ids)} val tokens.")
    print(f"  {tokens_path}")
    print(f"  {vocab_path}")


if __name__ == "__main__":
    main()
