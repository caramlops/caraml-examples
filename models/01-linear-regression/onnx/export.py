"""Exports the trained torch linear-regression model to ONNX.

Run models/01-linear-regression/torch/train.py first so torch/model.pt
exists.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "torch"))
from model import LinearRegressionModule

TORCH_MODEL_PATH = Path(__file__).parent.parent / "torch" / "model.pt"
ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"
N_FEATURES = 3


def main():
    model = LinearRegressionModule(n_features=N_FEATURES)
    model.load_state_dict(torch.load(TORCH_MODEL_PATH, weights_only=True))
    model.eval()

    dummy_input = torch.zeros(1, N_FEATURES)

    # TODO(you): export `model` to ONNX at ONNX_MODEL_PATH using
    # torch.onnx.export. Pass `dummy_input` as the example input, name the
    # input/output tensors "input"/"output", and mark the batch dimension
    # (axis 0 of both input and output) as dynamic via `dynamic_axes` so the
    # exported graph accepts any batch size at inference time:
    #   dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    raise NotImplementedError("Implement the torch.onnx.export call")

    print(f"Exported ONNX model to {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()
