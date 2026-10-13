"""Tiny synthetic dataset in Kaggle's folder layout, shared by the tests."""
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ("A___healthy", "B___rust", "C___scab")


def make_dataset(root: Path, n_orig=10, copies=3, seed=0, size=64):
    """Each 'original' texture gets `copies` slightly brightened copies, split naively 80/20 (= leaky)."""
    rng = np.random.default_rng(seed)
    base = root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
    for split in ("train", "valid"):
        for c in CLASSES:
            (base / split / c).mkdir(parents=True)
    (root / "test" / "test").mkdir(parents=True)
    for ci, c in enumerate(CLASSES):
        for o in range(n_orig):
            img = rng.integers(0, 255, (size // 8, size // 8, 3)).astype(np.uint8)
            img = np.kron(img, np.ones((8, 8, 1), np.uint8))
            img[:, :, ci] = np.clip(img[:, :, ci].astype(int) + 60, 0, 255)  # make classes weakly separable
            for k in range(copies):
                aug = np.clip(img.astype(int) + rng.integers(-3, 4), 0, 255).astype(np.uint8)
                split = "valid" if rng.random() < 0.2 else "train"
                Image.fromarray(aug).save(base / split / c / f"{c}_{o}_{k}.JPG")
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(root / "test" / "test" / "t1.JPG")
