"""Generates CIFAR-10-like images from pure noise via the full DDPM
reverse (sampling) process -- the actual "ships" deliverable for this
model. Calls the trained torch U-Net purely as a noise predictor (a black
box from numpy's point of view); every step of the actual sampling
process is implemented in diffusion.py and driven from here.

Run models/07-diffusion/torch/train.py first so torch/model.pt exists.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "torch"))
from diffusion import make_schedule, p_sample_step
from model import UNet

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

TORCH_MODEL_PATH = Path(__file__).parent.parent / "torch" / "model.pt"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

T = 300
BETA_START, BETA_END = 1e-4, 0.02
N_SAMPLES = 8

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def predict_noise(model: UNet, x_t: np.ndarray, t: int) -> np.ndarray:
    """The one place this file touches torch: run the trained network on
    the current (numpy) noisy batch and hand a plain numpy array of
    predicted noise back. Everything about *how* that prediction gets
    used to produce x_{t-1} lives in diffusion.py, not here."""
    with torch.no_grad():
        x_t_torch = torch.tensor(x_t, dtype=torch.float32, device=DEVICE)
        t_torch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=DEVICE)
        return model(x_t_torch, t_torch).cpu().numpy()


def save_samples(images: np.ndarray, path: Path) -> None:
    images = np.clip((images + 1.0) / 2.0, 0.0, 1.0)  # [-1, 1] -> [0, 1] for display
    fig, axes = plt.subplots(1, len(images), figsize=(2 * len(images), 2))
    for ax, img in zip(axes, images):
        ax.imshow(img.transpose(1, 2, 0))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(TORCH_MODEL_PATH, weights_only=True, map_location=DEVICE))
    model.eval()

    schedule = make_schedule(T, BETA_START, BETA_END)

    rng = np.random.default_rng(0)
    x_t = rng.normal(size=(N_SAMPLES, 3, 32, 32)).astype(np.float32)

    for t in reversed(range(T)):
        eps_pred = predict_noise(model, x_t, t)
        noise = rng.normal(size=x_t.shape).astype(np.float32) if t > 0 else np.zeros_like(x_t)
        x_t = p_sample_step(x_t, t, eps_pred, schedule, noise)
        if t % 50 == 0:
            log.info(f"sampling step t={t}")

    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / "samples.png"
    save_samples(x_t, out_path)
    log.info(f"Saved {N_SAMPLES} generated samples to {out_path}")


if __name__ == "__main__":
    main()
