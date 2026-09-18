"""Exports the trained torch ridge-regression model to ONNX.

Run models/02-ridge-regression/torch/train.py first so torch/model.pt
exists.
"""

import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "torch"))
from model import RidgeRegressionModule

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

TORCH_MODEL_PATH = Path(__file__).parent.parent / "torch" / "model.pt"
ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"
N_FEATURES = 5


def main():
    model = RidgeRegressionModule(n_features=N_FEATURES)
    model.load_state_dict(torch.load(TORCH_MODEL_PATH, weights_only=True))
    model.eval()

    dummy_input = torch.zeros(1, N_FEATURES)
    log.debug(f"dummy_input shape={dummy_input.shape}")

    torch.onnx.export(
        model,
        (dummy_input,),
        ONNX_MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
    )

    log.info(f"Exported ONNX model to {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()
