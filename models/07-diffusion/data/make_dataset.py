"""Downloads and unpacks CIFAR-10 (60,000 32x32 RGB images, 10 classes) --
real photographs, not synthetic data, because a diffusion model's entire
job is learning the structure of real images; there's nothing for it to
learn from synthetic noise-like data.

Sourced from a GitHub-hosted mirror of the individual JPEG files
(YoongiKim/CIFAR-10-images), not the dataset's official host
(cs.toronto.edu) -- that host rate-limits downloads to a few KB/s from
this environment's network (confirmed: ~1.9 KB/s vs. ~9 MB/s from this
mirror), which would take the better part of a day for the ~170MB
official pickled tarball. This mirror's plain JPEGs are also smaller
on the wire (~20MB, JPEG-compressed) and sidestep ever unpickling
untrusted data.

Pixels are scaled to [-1, 1] (not [0, 1]) -- the standard normalization
for DDPM, since the model predicts *noise* added in that range (see
PAPER.md) and the forward diffusion process assumes data living roughly
within it.

Fully implemented on purpose: the point of this model is deriving and
implementing the diffusion forward/reverse process and training the
denoising network, not data wrangling.
"""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parent
CORPUS_URL = "https://github.com/YoongiKim/CIFAR-10-images/archive/refs/heads/master.tar.gz"
TARBALL_PATH = DATA_DIR / "cifar10-images.tar.gz"
EXTRACTED_DIR = DATA_DIR / "CIFAR-10-images-master"
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def download_and_extract() -> None:
    if EXTRACTED_DIR.exists():
        return
    if not TARBALL_PATH.exists():
        print(f"Downloading corpus from {CORPUS_URL} ...")
        urllib.request.urlretrieve(CORPUS_URL, TARBALL_PATH)
    print("Extracting ...")
    with tarfile.open(TARBALL_PATH) as tf:
        tf.extractall(DATA_DIR, filter="data")


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = EXTRACTED_DIR / split / class_name
        for path in sorted(class_dir.glob("*.jpg")):
            images.append(np.array(Image.open(path)))
            labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.int64)


def main():
    download_and_extract()

    train_images, train_labels = load_split("train")
    test_images, test_labels = load_split("test")

    # Scale uint8 [0, 255] to float32 [-1, 1].
    train_images = (train_images.astype(np.float32) / 127.5) - 1.0
    test_images = (test_images.astype(np.float32) / 127.5) - 1.0

    npz_path = DATA_DIR / "cifar10.npz"
    np.savez(
        npz_path,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    print(
        f"Wrote {len(train_images)} train / {len(test_images)} test images, shape {train_images.shape[1:]}."
    )
    print(f"  {npz_path}")


if __name__ == "__main__":
    main()
