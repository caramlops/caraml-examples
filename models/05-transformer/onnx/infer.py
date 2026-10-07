import json
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"

BLOCK_SIZE = 64
N_NEW_TOKENS = 300
TEMPERATURE = 0.8


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    stoi, itos = vocab["stoi"], {int(k): v for k, v in vocab["itos"].items()}

    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    rng = np.random.default_rng(0)

    prompt = "ROMEO:"
    ids = np.array([stoi[ch] for ch in prompt], dtype=np.int64)
    log.debug(f"prompt ids shape={ids.shape}")

    # TODO(you): run inference with `session`, same pattern as
    # 01-linear-regression/onnx/infer.py, but in a generation loop:
    #   - input_name = session.get_inputs()[0].name
    #   - output_name = session.get_outputs()[0].name
    #   - Repeat N_NEW_TOKENS times:
    #       1. context = ids[-BLOCK_SIZE:][None, :] -- shape (1, T),
    #          the last BLOCK_SIZE (or fewer, early on) tokens.
    #       2. result = session.run([output_name], {input_name: context})
    #       3. result[0] has shape (1, T, vocab_size) -- raw logits for
    #          every position. You only want the *last* position's
    #          logits (the prediction for the next character):
    #          last_logits = result[0][0, -1] / TEMPERATURE
    #       4. Softmax last_logits by hand (subtract the max first, same
    #          numerically-stable pattern as every other runtime), then
    #          sample the next token id from that distribution with
    #          rng.choice(vocab_size, p=probs) -- not argmax, or you'll
    #          get the same token forever.
    #       5. Append the sampled id to `ids` and continue.
    #   - Decode the final `ids` back to text via itos and log.info(...) it.
    raise NotImplementedError("Implement the onnxruntime generation loop")


if __name__ == "__main__":
    main()
