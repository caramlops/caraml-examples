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


def main():
    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    sample = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
    log.debug(f"sample shape={sample.shape}")

    # TODO(you): run inference with `session`, same pattern as
    # 01-linear-regression/onnx/infer.py.
    #   - input_name = session.get_inputs()[0].name
    #   - output_name = session.get_outputs()[0].name
    #   - result = session.run([output_name], {input_name: sample})
    #   - result[0] is the raw linear predictor eta, not the mean count --
    #     apply np.exp() to it before logging.
    raise NotImplementedError("Implement the onnxruntime inference call")


if __name__ == "__main__":
    main()
