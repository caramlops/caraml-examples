import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    sample = np.array([[2.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    log.debug(f"sample shape={sample.shape}")

    # TODO(you): run inference with `session`, same pattern as
    # 01-linear-regression/onnx/infer.py.
    #   - input_name = session.get_inputs()[0].name
    #   - output_name = session.get_outputs()[0].name
    #   - result = session.run([output_name], {input_name: sample})
    #   - result[0] is raw logits, shape (1, n_classes) -- pass it through
    #     the _softmax() helper above, then argmax to get a class
    #     prediction. Log both with log.info(...).
    raise NotImplementedError("Implement the onnxruntime inference call")


if __name__ == "__main__":
    main()
