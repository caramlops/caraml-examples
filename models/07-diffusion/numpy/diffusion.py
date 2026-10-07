"""The DDPM forward (noising) and reverse (denoising/sampling) process
equations, implemented by hand -- the actual mathematical content of Ho
et al.'s paper. The noise-*prediction network* (torch/model.py's U-Net)
is a black box from this file's point of view: numpy/infer.py calls it to
get a noise prediction at each step, and everything here is the diffusion
*process* built around that prediction, which is what you're deriving.

Both pieces are placeholders because they're genuinely different
derivations: q_sample is a closed-form formula (no iteration involved);
p_sample_step is one iteration of the derived reverse-process posterior.
"""

import numpy as np


def make_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict[str, np.ndarray]:
    """Standard linear beta schedule (DDPM section 4) and its derived
    quantities. Fully implemented -- precomputed constants, not the
    paper's actual mathematical content."""
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
    }


def q_sample(x0: np.ndarray, t: int, noise: np.ndarray, schedule: dict[str, np.ndarray]) -> np.ndarray:
    """The forward diffusion process in closed form (DDPM eq. 4): sample
    x_t directly from x_0, without iterating through t individual noising
    steps. x0, noise: (B, 3, 32, 32). t: a single integer timestep shared
    across the batch (unlike training, inference/visualization typically
    noises a whole batch to the same t)."""
    # TODO(you): implement the closed-form forward process.
    #   x_t = sqrt_alphas_cumprod[t] * x0 + sqrt_one_minus_alphas_cumprod[t] * noise
    #   Both schedule values are scalars (indexing a 1-D array of length T
    #   with a single int t) -- they broadcast directly against x0/noise's
    #   (B, 3, 32, 32) shape with no reshaping needed.
    raise NotImplementedError("Implement the closed-form forward diffusion process")


def p_sample_step(
    x_t: np.ndarray, t: int, predicted_noise: np.ndarray, schedule: dict[str, np.ndarray], noise: np.ndarray
) -> np.ndarray:
    """One step of the reverse process (DDPM Algorithm 2 / eq. 11):
    given the current noisy image x_t and the network's prediction of the
    noise in it, compute x_{t-1}. x_t, predicted_noise, noise:
    (B, 3, 32, 32). `noise` is fresh random noise for the stochastic term
    -- the caller passes np.zeros_like(x_t) for it at t=0, since the very
    last step of sampling is deterministic (DDPM Algorithm 2, line 4)."""
    # TODO(you): implement one reverse-process step.
    #   alpha_t = schedule["alphas"][t]
    #   alpha_bar_t = schedule["alphas_cumprod"][t]
    #   beta_t = schedule["betas"][t]
    #
    #   mean = (1 / sqrt(alpha_t)) * (
    #       x_t - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise
    #   )
    #   sigma_t = sqrt(beta_t)   -- DDPM's simplest choice of reverse
    #       variance (section 3.2); the paper notes a couple of
    #       reasonable choices here, this is the one eq. 11's "simple"
    #       loss was derived to pair with.
    #   return mean + sigma_t * noise
    raise NotImplementedError("Implement one reverse-diffusion sampling step")
