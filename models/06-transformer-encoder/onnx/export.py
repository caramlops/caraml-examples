"""Exports the trained torch BERT-style classifier to ONNX.

Run models/06-transformer-encoder/torch/train.py first so torch/model.pt
exists.
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "torch"))
from model import BERTClassifier

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
TORCH_MODEL_PATH = Path(__file__).parent.parent / "torch" / "model.pt"
ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"] + 2
    max_len = vocab["max_len"] + 1

    model = BERTClassifier(
        vocab_size=vocab_size,
        max_len=max_len,
        n_classes=len(vocab["label_names"]),
        pad_id=vocab["pad_id"],
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    )
    model.load_state_dict(torch.load(TORCH_MODEL_PATH, weights_only=True, map_location="cpu"))
    model.eval()

    # Every real input is padded to exactly max_len (unlike 05's growing
    # generation context), so only the batch dimension needs to be
    # dynamic here -- sequence length is always fixed at max_len.
    dummy_input = torch.zeros(1, max_len, dtype=torch.long)
    log.debug(f"dummy_input shape={dummy_input.shape}")

    # TODO(you): export `model` to ONNX at ONNX_MODEL_PATH using
    # torch.onnx.export, same pattern as 01-linear-regression/onnx/export.py.
    #   - Pass `dummy_input` as the example input, name the input/output
    #     tensors "input"/"output".
    #   - Mark the batch dimension (axis 0) as dynamic:
    #       dynamic_shapes=({0: torch.export.Dim("batch")},)
    raise NotImplementedError("Implement the torch.onnx.export call")

    log.info(f"Exported ONNX model to {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()
