"""End-to-end test of scripts 01-03 on a tiny synthetic dataset in Kaggle's folder layout.

Random embedder weights are used (no download), so this checks the plumbing only, not the
quality of pretrained embeddings.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _make_dataset(root: Path, n_orig=10, copies=3, seed=0):
    rng = np.random.default_rng(seed)
    base = root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
    for split in ("train", "valid"):
        for c in ("A___healthy", "B___rust", "C___scab"):
            (base / split / c).mkdir(parents=True)
    (root / "test" / "test").mkdir(parents=True)
    for c in ("A___healthy", "B___rust", "C___scab"):
        for o in range(n_orig):
            img = rng.integers(0, 255, (32, 32, 3)).astype(np.uint8)
            img = np.kron(img, np.ones((8, 8, 1), np.uint8))               # blocky 256x256 texture
            for k in range(copies):
                aug = np.clip(img.astype(int) + rng.integers(-3, 4), 0, 255).astype(np.uint8)
                split = "valid" if rng.random() < 0.2 else "train"          # naive split of augmented files
                Image.fromarray(aug).save(base / split / c / f"{c}_{o}_{k}.JPG")
    Image.fromarray(rng.integers(0, 255, (256, 256, 3)).astype(np.uint8)).save(root / "test" / "test" / "t1.JPG")


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_scripts_end_to_end(tmp_path):
    data = tmp_path / "data"
    _make_dataset(data)
    art, res = tmp_path / "art", tmp_path / "res"
    _run("scripts/01_inspect_data.py", "--root", data, "--out", art / "inspect", "--results", res / "inspect.json")
    ins = json.loads((res / "inspect.json").read_text())["metrics"]
    assert ins["n_classes"] == 3 and ins["same_classes"]

    _run("scripts/02_leakage.py", "--root", data, "--out", art / "leak", "--results", res / "leak.json",
         "--no-pretrained", "--limit", 200, "--workers", 0)
    leak = json.loads((res / "leak.json").read_text())["metrics"]
    assert leak["valid_to_train"]["frac_at_least"]["0.9"] > 0.5      # naive split leaks heavily

    csv = tmp_path / "split.csv.gz"
    _run("scripts/03_clean_split.py", "--feats", art / "leak" / "feats_limit200.npz", "--threshold", 0.9,
         "--csv", csv, "--results", res / "clean.json")
    clean = json.loads((res / "clean.json").read_text())["metrics"]
    assert clean["clean_split"]["of_which_same_label"] == 0
    assert clean["original_split"]["fraction"] > clean["clean_split"]["val_with_near_copy_in_train"] / clean["clean_split"]["n_val"]
    df = pd.read_csv(csv)
    assert set(df.split) == {"train", "val"}
    assert (df.groupby("group").split.nunique() == 1).all()          # no group straddles the split
