import json
import logging
import os
import sys
from pathlib import Path

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
N_NEW_TOKENS = 300
TEMPERATURE = 0.8

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(vocab_size: int) -> GPT:
    model = GPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def generate(model: GPT, prompt_ids: torch.Tensor, n_new_tokens: int) -> torch.Tensor:
    ids = prompt_ids.clone()
    for _ in range(n_new_tokens):
        context = ids[-model.block_size :].unsqueeze(0)
        logits = model(context)
        last_logits = logits[0, -1] / TEMPERATURE
        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id])
    return ids


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    stoi, itos = vocab["stoi"], {int(k): v for k, v in vocab["itos"].items()}

    model = load_model(vocab["vocab_size"])

    prompt = "ROMEO:"
    prompt_ids = torch.tensor([stoi[ch] for ch in prompt], device=DEVICE)
    generated_ids = generate(model, prompt_ids, N_NEW_TOKENS)
    text = "".join(itos[i] for i in generated_ids.tolist())
    log.info(f"Generated:\n{text}")


if __name__ == "__main__":
    main()
