"""Tiny synthetic datasets in Kaggle's folder layout, shared by the tests."""
import json
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


def _uuid(i: int) -> str:
    return f"{i:08x}-1111-2222-3333-{i:012x}"


def make_grouped_dataset(root: Path, n_sources=40, seed=0, size=64):
    """PlantVillage-like names: `<uuid>___RS_HL <n>[_aug].JPG`, ' copy' twins with their OWN uuid, and class C without any
    uuid (like Corn_Common_rust_). Files are dropped randomly into train/valid (a leaky original split).
    Returns {leaf_map: dict} ready to be dumped as the authors' leaf-map.json."""
    rng = np.random.default_rng(seed)
    base = root / "d" / "d"   # short folder names: UUID file names are long and Windows paths are capped at 260 chars
    for split in ("train", "valid"):
        for c in CLASSES:
            (base / split / c).mkdir(parents=True)
    (root / "test" / "test").mkdir(parents=True)
    augs = ["", "_flipLR", "_90deg", "_new30degFlipLR"]
    leaf_map, uid = {}, 1000
    for ci, c in enumerate(CLASSES):
        for s in range(n_sources):
            num = 100 + s
            img = rng.integers(0, 255, (size // 8, size // 8, 3)).astype(np.uint8)
            img = np.kron(img, np.ones((8, 8, 1), np.uint8))
            img[:, :, ci] = np.clip(img[:, :, ci].astype(int) + 60, 0, 255)
            variants = [(f"RS_HL {num}", _uuid(uid))]
            uid += 1
            if c != "C___scab" and s % 5 == 0:                     # a ' copy' twin: same picture, different uuid
                variants.append((f"RS_HL {num} copy", _uuid(uid)))
                uid += 1
            for body, u in variants:
                for a in rng.permutation(augs)[:int(rng.integers(2, 4))]:
                    name = (f"{u}___{body}{a}.JPG" if c != "C___scab" else f"RS_Rust {num}{a}.JPG")
                    Image.fromarray(img).save(base / rng.choice(["train", "valid"]) / c / name)
            if s % 10 < 7:                                          # ~70% of sources are known to the leaf map
                leaf_map.setdefault(f"rs_hl {num}" if c != "C___scab" else f"rs_rust {num}", []).append(f"{c}:::{s // 3}.0")
    leaf_map["rs_hl 100"] = leaf_map.get("rs_hl 100", []) + ["Z___other:::9.0"]   # same stem under another class: must be ignored
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(root / "test" / "test" / "t1.JPG")
    return leaf_map


def write_leaf_map(leaf_map: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(leaf_map))
    return path
