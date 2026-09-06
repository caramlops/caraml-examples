import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import LinearRegressionOLS

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent / "model.json"


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    feature_columns = [c for c in train_df.columns if c != "y"]

    model = LinearRegressionOLS(feature_columns)
    model.fit(train_df)

    preds = model.predict(test_df)
    mse = ((preds - test_df["y"]) ** 2).mean()
    print(f"weights=\n{model.weights}\nbias={model.bias}")
    print(f"Test MSE: {mse:.4f}")

    with open(MODEL_PATH, "w") as f:
        json.dump({"weights": model.weights.to_dict(), "bias": model.bias}, f)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
