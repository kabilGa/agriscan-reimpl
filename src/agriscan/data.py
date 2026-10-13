"""Locate and list the plant-disease images without hard-coding Kaggle's folder nesting."""
import os
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png"}


def _images_in(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]


def find_splits(root: str | Path) -> dict[str, Path]:
    """Find the train / valid (class-folder layout) and test (flat) directories under root.

    We search instead of assuming a path because we have not seen the exact nesting
    of the Kaggle download (it is reported to repeat folder names).
    """
    found: dict[str, Path] = {}
    for cur, dirs, _ in os.walk(root):
        cur = Path(cur)
        if cur.name in ("train", "valid") and cur.name not in found:
            kids = [cur / d for d in dirs]
            if len(kids) >= 2 and all(_images_in(k) for k in kids[:2]):
                found[cur.name] = cur
                dirs.clear()  # no need to descend into class folders
        elif cur.name == "test" and "test" not in found and _images_in(cur):
            found["test"] = cur
    missing = {"train", "valid"} - found.keys()
    if missing:
        raise FileNotFoundError(f"Could not find {sorted(missing)} under {root}")
    return found


def list_images(split_dir: Path, split: str, root: Path) -> pd.DataFrame:
    """One row per image: absolute path, path relative to root, class label, split name."""
    rows = []
    if split == "test":  # flat folder, no class subfolders
        files = [(p, "") for p in _images_in(split_dir)]
    else:
        files = [(p, c.name) for c in sorted(split_dir.iterdir()) if c.is_dir() for p in _images_in(c)]
    for p, label in sorted(files):
        rows.append((str(p), p.relative_to(root).as_posix(), label, split))
    return pd.DataFrame(rows, columns=["path", "rel", "label", "split"])
