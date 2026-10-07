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

T = 300  # diffusion timesteps -- scaled down from the paper's 1000 to keep
# local training/sampling tractable (see PAPER.md's scoped-down experiment)
BETA_START, BETA_END = 1e-4, 0.02
BATCH_SIZE = 64
N_ITERS = 3000
LR = 2e-4

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def make_schedule(T: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    """Standard linear beta schedule (DDPM section 4) and its derived
    quantities -- fully implemented, not paper-novel math itself, just
    precomputed constants the placeholder below uses."""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas.to(DEVICE),
        "alphas": alphas.to(DEVICE),
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod).to(DEVICE),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod).to(DEVICE),
    }


def get_batch(images: np.ndarray, batch_size: int) -> torch.Tensor:
    idx = np.random.randint(0, len(images), size=batch_size)
    batch = images[idx].transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return torch.tensor(batch, device=DEVICE)


def main():
    log.info(f"Using device: {DEVICE}")
    data = np.load(DATA_PATH)
    train_images = data["train_images"]
    log.debug(f"train_images shape={train_images.shape}")

    schedule = make_schedule(T, BETA_START, BETA_END)
    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(N_ITERS):
        x0 = get_batch(train_images, BATCH_SIZE)

        # TODO(you): implement one DDPM training step (Algorithm 1 in
        # Ho et al.) -- sample a random timestep and noise, construct the
        # noisy image in closed form, predict the noise, and take a
        # gradient step toward predicting it correctly.
        #   1. t = torch.randint(0, T, (x0.shape[0],), device=DEVICE) --
        #      a different random timestep per image in the batch.
        #   2. noise = torch.randn_like(x0)
        #   3. Forward process in closed form (DDPM eq. 4): rather than
        #      iterating t noising steps one at a time, you can sample
        #      x_t directly from x_0:
        #        x_t = sqrt_alphas_cumprod[t] * x0 +
        #              sqrt_one_minus_alphas_cumprod[t] * noise
        #      `schedule["sqrt_alphas_cumprod"]` has shape (T,); index it
        #      with `t` (shape (B,)) to get one scalar per image, then
        #      reshape to (B, 1, 1, 1) so it broadcasts against x0's
        #      (B, 3, 32, 32).
        #   4. predicted_noise = model(x_t, t)
        #   5. loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        #      -- DDPM's simplified objective (eq. 14): predict the exact
        #      noise that was added, nothing fancier.
        #   6. optimizer.zero_grad(); loss.backward(); optimizer.step()
        # Log every 200 steps: log.info(f"step {step}: loss {loss.item():.4f}")
        raise NotImplementedError("Implement one DDPM training step")

    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
