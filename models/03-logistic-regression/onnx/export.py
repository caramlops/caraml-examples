"""Exports the trained torch logistic-regression model to ONNX.

Run models/03-logistic-regression/torch/train.py first so torch/model.pt
exists.
"""

import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "torch"))
from model import LogisticRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

TORCH_MODEL_PATH = Path(__file__).parent.parent / "torch" / "model.pt"
ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"
N_FEATURES = 4
N_CLASSES = 3


def main():
    model = LogisticRegressionModule(n_features=N_FEATURES, n_classes=N_CLASSES)
    model.load_state_dict(torch.load(TORCH_MODEL_PATH, weights_only=True))
    model.eval()

    dummy_input = torch.zeros(1, N_FEATURES)
    log.debug(f"dummy_input shape={dummy_input.shape}")

    # TODO(you): export `model` to ONNX at ONNX_MODEL_PATH using
    # torch.onnx.export, same pattern as
    # 01-linear-regression/onnx/export.py.
    #   - Pass `dummy_input` as the example input, name the input/output
    #     tensors "input"/"output".
    #   - Mark the batch dimension (axis 0) as dynamic:
    #       dynamic_shapes=({0: torch.export.Dim("batch")},)
    #   - The model's output is raw logits, shape (batch, n_classes), not
    #     probabilities -- onnx/infer.py applies softmax + argmax itself
    #     after running the session, same division of labor as the torch
    #     runtime's forward pass vs. its loss function.
    raise NotImplementedError("Implement the torch.onnx.export call")

    log.info(f"Exported ONNX model to {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()
