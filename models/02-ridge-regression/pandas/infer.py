import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import RidgeRegression

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.json"


def main():
    with open(MODEL_PATH) as f:
        params = json.load(f)

    feature_columns = list(params["weights"].keys())
    model = RidgeRegression(feature_columns)
    model.weights = pd.Series(params["weights"])
    model.bias = params["bias"]

    sample = pd.DataFrame([[1.0, -1.0, 0.5, 0.0, 0.0]], columns=feature_columns)
    prediction = model.predict(sample)
    log.info(f"Prediction for {sample.values.tolist()}: {prediction.tolist()}")


if __name__ == "__main__":
    main()
