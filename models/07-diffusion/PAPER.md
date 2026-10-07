# Denoising Diffusion Probabilistic Models (DDPM)

Jonathan Ho, Ajay Jain, Pieter Abbeel. "Denoising Diffusion Probabilistic
Models." NeurIPS 2020. https://arxiv.org/abs/2006.11239

## The core idea

Train a network to do one very narrow thing well — given a somewhat-noisy
image and how far along the noising process it is, predict exactly what
noise was added — and it turns out you can use that network to *generate*
entirely new images: start from pure random noise and repeatedly ask the
network "what noise is in this?", subtract an appropriately-scaled amount
of its answer, and after enough steps you're left with something that
looks like a sample from the training distribution. The "forward process"
(systematically destroying an image with noise over `T` steps) is fixed
and has no learned parameters at all; only the "reverse process" (the
noise-prediction network) is trained.

## Key equations reproduced here

**Forward process, closed form** (eq. 4) — the paper defines the forward
process as a Markov chain of `T` individual noising steps, but shows you
never need to actually iterate through them to get `x_t` from `x_0`:

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,   epsilon ~ N(0, I)
```

where `alpha_bar_t` is the cumulative product of `(1 - beta_s)` for
`s = 1..t`, and `beta_1..beta_T` is a fixed, non-learned noise schedule
(section 4: linear from `1e-4` to `0.02`).

**Training objective, simplified** (eq. 14) — rather than optimizing the
full variational bound, the paper shows a simpler objective works just as
well in practice: train a network `epsilon_theta(x_t, t)` to predict the
exact noise that was added,

```
L_simple = E[ || epsilon - epsilon_theta(x_t, t) ||^2 ]
```

**Reverse process, one sampling step** (Algorithm 2 / eq. 11) — given
`x_t` and the network's noise prediction, compute `x_{t-1}`:

```
x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z
```

where `z ~ N(0, I)` for every step except the very last (`t=0`, where it's
deterministic), and `sigma_t = sqrt(beta_t)` is the paper's simplest
choice of reverse-process variance.

## What's a deliberate deviation from the paper

- **`T = 300`, not `T = 1000`.** Fewer diffusion steps means a coarser
  noise schedule and somewhat lower sample quality, but roughly
  proportionally faster training and (especially) sampling — sampling
  requires one full network forward pass *per timestep*, so this is a 3.3x
  speedup on the single slowest part of actually seeing this work locally.
- **A small U-Net** (two downsampling stages, 64/128/128 channels, one
  residual block per resolution) instead of the paper's deeper
  architecture with self-attention layers at low resolutions. This is
  purely a capacity/speed tradeoff, not a conceptual simplification — the
  same time-conditioned residual-block structure, just smaller.

## Scoped-down experiment

The paper trains on CIFAR-10 (among other datasets) for 800k-1M steps on
TPU hardware. This reproduces the same dataset — real photographs, not
synthetic data, since a diffusion model's entire job is learning the
structure of real images — but trains for a few thousand steps locally.
That's nowhere near enough for the sharp, recognizable samples the paper
reports; the realistic goal here is confirming the *mechanism* works
(loss decreasing, generated samples showing color/structure rather than
pure noise), not matching the paper's reported sample quality or FID
score, which needs orders of magnitude more compute.

## Metric / result to compare against

The paper reports Inception Score and FID on CIFAR-10 after full training
— not meaningfully reproducible at this scale. The realistic signal to
look for here: training loss (noise-prediction MSE) should drop from
around `1.0` (the "predict all-zero noise" baseline, since the added
noise is standard normal) into a noticeably lower range within a few
thousand steps, and `numpy/infer.py`'s generated samples should show
blob-like color structure rather than remaining visually indistinguishable
from the pure-noise starting point.
