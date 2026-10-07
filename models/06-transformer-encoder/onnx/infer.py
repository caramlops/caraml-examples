import json
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
ONNX_MODEL_PATH = Path(__file__).parent / "model.onnx"


def encode(message: str, stoi: dict[str, int], max_len: int) -> list[int]:
    ids = [stoi["[CLS]"]] + [stoi[ch] for ch in message[: max_len - 1]]
    ids += [stoi["[PAD]"]] * (max_len - len(ids))
    return ids


def main():
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    max_len = vocab["max_len"] + 1

    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    message = "FREE entry! Win a brand new car, text WIN to 80085 now!!!"
    sample = np.array([encode(message, vocab["stoi"], max_len)], dtype=np.int64)
    log.debug(f"sample shape={sample.shape}")

    # TODO(you): run inference with `session`, same pattern as
    # 01-linear-regression/onnx/infer.py.
    #   - input_name = session.get_inputs()[0].name
    #   - output_name = session.get_outputs()[0].name
    #   - result = session.run([output_name], {input_name: sample})
    #   - result[0] is raw logits, shape (1, n_classes) -- softmax it by
    #     hand (subtract the max first, same pattern as every other
    #     runtime) and argmax to get the predicted label. Use
    #     vocab["label_names"][predicted_index] to turn that back into
    #     "ham"/"spam", and log.info(...) the message, label, and probs.
    raise NotImplementedError("Implement the onnxruntime inference call")


if __name__ == "__main__":
    main()
