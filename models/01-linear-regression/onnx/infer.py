from pathlib import Path

import numpy as np
import onnxruntime as ort

ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"


def main():
    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    sample = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)

    # TODO(you): run inference with `session`.
    #   Call session.run(output_names, input_feed) where output_names=None
    #   returns every output, and input_feed maps the input tensor's name
    #   (session.get_inputs()[0].name) to `sample`. Print the prediction.
    raise NotImplementedError("Implement the onnxruntime inference call")


if __name__ == "__main__":
    main()
