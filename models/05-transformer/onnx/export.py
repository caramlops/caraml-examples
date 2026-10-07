"""Exports the trained torch transformer to ONNX.

Run models/05-transformer/torch/train.py first so torch/model.pt exists.
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "torch"))
from model import GPT

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
TORCH_MODEL_PATH = Path(__file__).parent.parent / "torch" / "model.pt"
ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"

BLOCK_SIZE = 64
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)

    model = GPT(
        vocab_size=vocab["vocab_size"],
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
    )
    model.load_state_dict(torch.load(TORCH_MODEL_PATH, weights_only=True, map_location="cpu"))
    model.eval()

    dummy_input = torch.zeros(1, BLOCK_SIZE, dtype=torch.long)
    log.debug(f"dummy_input shape={dummy_input.shape}")

    # TODO(you): export `model` to ONNX at ONNX_MODEL_PATH using
    # torch.onnx.export, same pattern as 01-linear-regression/onnx/export.py
    # -- but with TWO dynamic dimensions this time, not just the batch.
    #   - Pass `dummy_input` as the example input, name the input/output
    #     tensors "input"/"output".
    #   - Generation calls the model with a growing context length each
    #     step (from infer.py's sampling loop), not just varying batch
    #     sizes -- so both axis 0 (batch) AND axis 1 (sequence length)
    #     need to be dynamic:
    #       batch = torch.export.Dim("batch")
    #       seq = torch.export.Dim("seq", max=BLOCK_SIZE)
    #       dynamic_shapes=({0: batch, 1: seq},)
    #     (seq needs an explicit max= because the model's positional
    #     encoding table only has BLOCK_SIZE rows -- the exporter needs to
    #     know the valid range, not just that the dimension varies at all.)
    raise NotImplementedError("Implement the torch.onnx.export call")

    log.info(f"Exported ONNX model to {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()
