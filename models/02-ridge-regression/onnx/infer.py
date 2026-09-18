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
    sample = np.array([[1.0, -1.0, 0.5, 0.0, 0.0]], dtype=np.float32)
    log.debug(f"sample shape={sample.shape}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: sample})

    log.info(f"Prediction for {sample.tolist()}: {result[0].tolist()}")


if __name__ == "__main__":
    main()
