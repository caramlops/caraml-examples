# 07 · Diffusion (DDPM, CIFAR-10)

Generate CIFAR-10-like images from pure noise via iterative denoising. See
`PAPER.md` for the paper this reproduces (Ho et al.'s DDPM), the exact
equations, and what was scoped down to make this trainable locally.

The data is real CIFAR-10 photographs (`data/make_dataset.py` downloads
and unpacks it, no `torchvision` dependency — just the official pickled
tarball parsed directly) — a diffusion model's entire job is learning the
structure of real images, so there's nothing for synthetic data to teach
it.

| Runtime | File(s) | Placeholder(s) | Fully implemented |
|---|---|---|---|
| torch | `torch/train.py` | one DDPM training step: sample `t`/noise, construct `x_t` in closed form, predict the noise, MSE loss, backward/step | the U-Net architecture (`torch/model.py` — standard building blocks, not this paper's contribution), data load, save/load |
| numpy | `numpy/diffusion.py` | the forward noising process (`q_sample`) and one reverse sampling step (`p_sample_step`) — genuinely different formulas, hence two placeholders | the full sampling loop (`numpy/infer.py`), schedule precomputation |

**No tensorflow, scipy, or pandas runtime for this model**, and torch's
role is narrower than usual too — see "Why only two runtimes" below.

## Why only two runtimes, and why they split the work this way

A diffusion U-Net genuinely needs GPU-accelerated autodiff to train in any
practical amount of time — reimplementing that by hand in numpy (manual
backprop through a multi-stage CNN with skip connections and time
conditioning) would be an enormous undertaking with little new to learn
beyond what `05-transformer/numpy`'s manual-autograd exercise already
covers, and wouldn't actually finish training locally either way. So
**torch is the only runtime that trains**, and tensorflow isn't
duplicated here the way it was for every earlier model — this is the one
place in the repo where that comparison isn't worth the overhead.

numpy's role is narrower and more targeted: it implements the **diffusion
math itself** (the forward noising formula and the reverse sampling
update rule — the paper's actual mathematical contribution), calling the
*trained* torch network purely as a noise-prediction black box
(`numpy/infer.py`'s `predict_noise()` is the one place it touches torch at
all). That split — torch owns the network and its training, numpy owns
the process built around it — is deliberate: it's the division of labor
that was actually practical to build and verify, and it still means you
derive and implement the real DDPM equations by hand, just without also
hand-deriving CNN backprop.

## Running it

```bash
# once (downloads + unpacks CIFAR-10, ~170MB):
uv run python models/07-diffusion/data/make_dataset.py

# train the U-Net (implement torch/train.py's placeholder first):
uv run python models/07-diffusion/torch/train.py

# sanity-check the trained denoiser (fully implemented, no placeholder):
uv run python models/07-diffusion/torch/infer.py

# generate images via the full reverse process (implement
# numpy/diffusion.py's two placeholders first):
uv run python models/07-diffusion/numpy/infer.py
```

`numpy/infer.py` writes generated samples to `plots/samples.png`. Don't
expect sharp, recognizable CIFAR-10 photos at the training budget this
repo can realistically run locally (a few thousand steps, vs. the paper's
~1M) — the realistic signal is training loss dropping meaningfully below
the ~1.0 predict-zero baseline, and generated samples showing blob-like
color structure rather than staying visually indistinguishable from pure
noise. See `PAPER.md`'s "Metric / result to compare against" for the
honest bar to check against.
