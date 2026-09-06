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

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "model.json"
ALPHA = 5.0


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    feature_columns = [c for c in train_df.columns if c != "y"]
    log.debug(f"train_df shape={train_df.shape}, test_df shape={test_df.shape}")

    model = RidgeRegression(feature_columns, alpha=ALPHA)
    model.fit(train_df)

    preds = model.predict(test_df)
    mse = ((preds - test_df["y"]) ** 2).mean()
    log.info(f"weights=\n{model.weights}\nbias={model.bias}")
    log.info(f"Test MSE: {mse:.4f}")

    with open(MODEL_PATH, "w") as f:
        json.dump({"weights": model.weights.to_dict(), "bias": model.bias}, f)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
