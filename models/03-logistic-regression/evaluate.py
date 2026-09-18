"""Confusion matrix + calibration (reliability) plot for the trained
logistic regression model.

Run models/03-logistic-regression/numpy/train.py first so numpy/model.npz
exists. This script always evaluates the numpy runtime's saved model
specifically -- every runtime fits the same data to the same objective, so
there's nothing to be learned by re-plotting the same two charts six times.

Fully implemented: this is analysis/reporting on top of the model you
already implemented, not a new placeholder.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "numpy"))
from model import LogisticRegression

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "logistic_regression.npz"
MODEL_PATH = Path(__file__).parent / "numpy" / "model.npz"
PLOTS_DIR = Path(__file__).parent / "plots"
N_CALIBRATION_BINS = 10


def load_model() -> LogisticRegression:
    params = np.load(MODEL_PATH)
    n_features, n_classes = params["weights"].shape
    model = LogisticRegression(n_features=n_features, n_classes=n_classes)
    model.weights = params["weights"]
    model.bias = params["bias"]
    return model


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    n_classes = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(confidences: np.ndarray, correct: np.ndarray, path: Path) -> None:
    # Reliability diagram: bin predictions by their top-1 confidence, and
    # compare each bin's mean confidence against its actual accuracy. A
    # perfectly calibrated model's points sit on the y=x diagonal --
    # points below it mean the model is overconfident in that confidence
    # range, points above mean it's underconfident.
    bin_edges = np.linspace(0, 1, N_CALIBRATION_BINS + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1])

    mean_confidences, accuracies, counts = [], [], []
    for b in range(N_CALIBRATION_BINS):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_confidences.append(confidences[mask].mean())
        accuracies.append(correct[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.scatter(mean_confidences, accuracies, s=[20 + 4 * c for c in counts], label="model")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration (reliability diagram)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    data = np.load(DATA_PATH)
    X_test, y_test = data["X_test"], data["y_test"]
    n_classes = int(data["n_classes"])

    model = load_model()
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    accuracy = (preds == y_test).mean()
    log.info(f"Test accuracy: {accuracy:.4f}")

    matrix = confusion_matrix(y_test, preds, n_classes)
    log.info(f"Confusion matrix:\n{matrix}")

    PLOTS_DIR.mkdir(exist_ok=True)
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(matrix, cm_path)
    log.info(f"Saved confusion matrix plot to {cm_path}")

    confidences = probs.max(axis=1)
    correct = (preds == y_test).astype(float)
    cal_path = PLOTS_DIR / "calibration.png"
    plot_calibration(confidences, correct, cal_path)
    log.info(f"Saved calibration plot to {cal_path}")


if __name__ == "__main__":
    main()
