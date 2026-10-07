"""Downloads the SMS Spam Collection dataset (5,574 real text messages,
labeled ham/spam) and tokenizes it at the character level, prepending a
[CLS] token to each sequence (its final-layer representation is what the
classification head reads, the same pooling trick BERT uses) and padding/
truncating every sequence to a fixed length so they can be batched.

A real, labeled dataset is used deliberately here (see CLAUDE.md's data
convention): a classifier trained on synthetic text-label pairs wouldn't
tell you anything about whether the implementation can actually learn a
genuine text-classification signal.

Fully implemented on purpose: the point of this model is deriving and
implementing bidirectional self-attention (with a padding mask, not a
causal one), not data wrangling.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
CORPUS_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
CORPUS_PATH = DATA_DIR / "sms.tsv"
SEED = 42
VAL_FRACTION = 0.15
MAX_LEN = 128  # characters, not counting [CLS]; covers ~90th percentile of message lengths


def download_corpus() -> list[tuple[str, str]]:
    if not CORPUS_PATH.exists():
        print(f"Downloading corpus from {CORPUS_URL} ...")
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
    rows = []
    for line in CORPUS_PATH.read_text().splitlines():
        label, message = line.split("\t", 1)
        rows.append((label, message))
    return rows


def build_vocab(messages: list[str]) -> dict[str, int]:
    chars = sorted(set("".join(messages)))
    # Character ids 0..len(chars)-1, then two special tokens appended:
    # [CLS] (prepended to every sequence) and [PAD] (fills out short
    # sequences to MAX_LEN + 1).
    stoi = {ch: i for i, ch in enumerate(chars)}
    stoi["[CLS]"] = len(chars)
    stoi["[PAD]"] = len(chars) + 1
    return stoi


def encode(message: str, stoi: dict[str, int]) -> np.ndarray:
    ids = [stoi["[CLS]"]] + [stoi[ch] for ch in message[:MAX_LEN]]
    ids += [stoi["[PAD]"]] * (MAX_LEN + 1 - len(ids))
    return np.array(ids, dtype=np.int64)


def main():
    rows = download_corpus()
    labels_str = [label for label, _ in rows]
    messages = [message for _, message in rows]

    stoi = build_vocab(messages)
    itos = {i: ch for ch, i in stoi.items()}
    label_names = sorted(set(labels_str))  # ["ham", "spam"]
    label_to_id = {name: i for i, name in enumerate(label_names)}

    X = np.stack([encode(m, stoi) for m in messages])
    y = np.array([label_to_id[label] for label in labels_str], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    npz_path = DATA_DIR / "sms.npz"
    np.savez(
        npz_path,
        train_ids=X[train_idx],
        train_labels=y[train_idx],
        val_ids=X[val_idx],
        val_labels=y[val_idx],
    )

    vocab_path = DATA_DIR / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(
            {
                "stoi": stoi,
                "itos": itos,
                "vocab_size": len(stoi) - 2,  # character vocab only, special tokens handled separately
                "cls_id": stoi["[CLS]"],
                "pad_id": stoi["[PAD]"],
                "max_len": MAX_LEN,
                "label_names": label_names,
            },
            f,
        )

    print(f"{len(X)} messages, {len(stoi) - 2} unique characters, labels: {label_names}")
    print(f"Wrote {len(train_idx)} train / {len(val_idx)} val rows.")
    print(f"  {npz_path}")
    print(f"  {vocab_path}")


if __name__ == "__main__":
    main()
