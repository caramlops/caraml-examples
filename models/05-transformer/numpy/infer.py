import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT

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
N_NEW_TOKENS = 300
TEMPERATURE = 0.8


def load_model() -> GPT:
    data = np.load(MODEL_PATH)
    vocab_size, block_size = int(data["vocab_size"]), int(data["block_size"])
    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    )
    params = model.params()
    for name in params:
        params[name][...] = data[name.replace(".", "__")]
    return model


def generate(model: GPT, prompt_ids: np.ndarray, n_new_tokens: int, rng: np.random.Generator) -> np.ndarray:
    ids = prompt_ids.copy()
    for _ in range(n_new_tokens):
        context = ids[-model.block_size :][None, :]  # (1, T)
        logits, _ = model.forward(context)
        last_logits = logits[0, -1] / TEMPERATURE
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum()
        next_id = rng.choice(len(probs), p=probs)
        ids = np.append(ids, next_id)
    return ids


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    stoi, itos = vocab["stoi"], {int(k): v for k, v in vocab["itos"].items()}

    model = load_model()
    rng = np.random.default_rng(0)

    prompt = "ROMEO:"
    prompt_ids = np.array([stoi[ch] for ch in prompt])
    generated_ids = generate(model, prompt_ids, N_NEW_TOKENS, rng)
    text = "".join(itos[i] for i in generated_ids)
    log.info(f"Generated:\n{text}")


if __name__ == "__main__":
    main()
