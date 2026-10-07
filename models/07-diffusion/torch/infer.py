"""A minimal sanity check for the trained denoiser -- NOT full image
generation. Full image sampling (the DDPM reverse process) is
numpy/infer.py's job (see README.md): this model's scoping deliberately
puts the iterative sampling *math* in numpy as the placeholder to derive,
using this trained network purely as a noise-prediction black box. This
script just confirms the trained network can predict noise reasonably
well on one batch, before you trust it inside that sampling loop.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import UNet

logging.basicConfig(
    level=os.environ.get("CARAML_LOGLEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "cifar10.npz"
MODEL_PATH = Path(__file__).parent / "model.pt"

T = 300
BETA_START, BETA_END = 1e-4, 0.02

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def main():
    data = np.load(DATA_PATH)
    x0 = data["test_images"][:8].transpose(0, 3, 1, 2)
    x0 = torch.tensor(x0, device=DEVICE)

    betas = torch.linspace(BETA_START, BETA_END, T, device=DEVICE)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))
    model.eval()

    t = torch.full((x0.shape[0],), T // 2, device=DEVICE, dtype=torch.long)
    noise = torch.randn_like(x0)
    sqrt_ac = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_1m_ac = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    x_t = sqrt_ac * x0 + sqrt_1m_ac * noise

    with torch.no_grad():
        predicted_noise = model(x_t, t)
    mse = torch.mean((predicted_noise - noise) ** 2).item()
    # noise ~ N(0, I), so predicting all-zero noise (i.e. a network that
    # learned nothing) gives MSE = Var(noise) = 1.0 -- the baseline a
    # working denoiser should land meaningfully below.
    log.info(f"Noise-prediction MSE at t={T // 2}: {mse:.4f} (predict-zero baseline is ~1.0)")


if __name__ == "__main__":
    main()
