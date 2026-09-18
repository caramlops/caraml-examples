import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import LogisticRegression

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "model.json"


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    feature_columns = [c for c in train_df.columns if c != "y"]
    n_classes = train_df["y"].nunique()
    log.debug(f"train_df shape={train_df.shape}, n_classes={n_classes}")

    model = LogisticRegression(feature_columns, n_classes=n_classes)
    model.fit(train_df)

    preds = model.predict(test_df)
    accuracy = (preds == test_df["y"]).mean()
    log.info(f"weights=\n{model.weights}\nbias=\n{model.bias}")
    log.info(f"Test accuracy: {accuracy:.4f}")

    with open(MODEL_PATH, "w") as f:
        json.dump({"weights": model.weights.values.tolist(), "bias": model.bias.values.tolist()}, f)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
