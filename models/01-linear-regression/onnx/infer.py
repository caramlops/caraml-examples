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

    # TODO(you): run inference with `session`.
    #   Call session.run(output_names, input_feed) where output_names=None
    #   returns every output, and input_feed maps the input tensor's name
    #   (session.get_inputs()[0].name) to `sample`. Log the prediction with
    #   log.info(...).
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    log.info(f"Input name, output name: {(input_name, output_name)}")

    result = session.run([output_name], {input_name: sample})

    log.info(f"Prediction for {sample.tolist()}: {result[0].tolist()}")


if __name__ == "__main__":
    main()
