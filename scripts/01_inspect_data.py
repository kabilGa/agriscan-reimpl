"""Step 1a: look at the dataset before trusting it (counts, image sizes, samples)."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from agriscan.data import find_splits, list_images
from agriscan.utils import save_results, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="dataset root, e.g. /kaggle/input/new-plant-diseases-dataset")
ap.add_argument("--out", default="artifacts/inspect")
ap.add_argument("--results", default="results/step1_inspect.json")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

set_seed(args.seed)
root = Path(args.root)
dirs = find_splits(root)
print("Found:", {k: str(v) for k, v in dirs.items()})
df = {s: list_images(d, s, root) for s, d in dirs.items()}

counts = {s: d[d.label != ""].label.value_counts().sort_index() for s, d in df.items() if s != "test"}
table = counts["train"].to_frame("train").join(counts["valid"].to_frame("valid"), how="outer")
print(table.to_string())
print({s: len(d) for s, d in df.items()})

same_classes = set(counts["train"].index) == set(counts["valid"].index)
print("n_classes:", len(table), "| same classes in train and valid:", same_classes)
print("train class size min/max:", counts["train"].min(), counts["train"].max())

# Sample sizes and readability. A broken file would crash training hours into a run.
rng = np.random.default_rng(args.seed)
sizes, bad = Counter(), []
for s in ("train", "valid"):
    for p in rng.choice(df[s].path.to_numpy(), size=min(1000, len(df[s])), replace=False):
        try:
            with Image.open(p) as im:
                im.load()
                sizes[im.size] += 1
        except Exception as e:  # noqa: BLE001
            bad.append((p, str(e)))
print("image sizes (2000 sampled):", dict(sizes), "| unreadable:", len(bad))

# One random image from 12 random classes, so we can eyeball what the data looks like.
Path(args.out).mkdir(parents=True, exist_ok=True)
classes = rng.choice(table.index.to_numpy(), size=min(12, len(table)), replace=False)
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax in axes.ravel():
    ax.axis("off")
for ax, c in zip(axes.ravel(), classes):
    p = rng.choice(df["train"][df["train"].label == c].path.to_numpy())
    ax.imshow(Image.open(p))
    ax.set_title(c, fontsize=7)
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{args.out}/samples.png", dpi=100)

save_results(args.results, name="step1_inspect", seed=args.seed, config={"root": str(root)},
             metrics={"n_images": {s: len(d) for s, d in df.items()},
                      "n_classes": len(table), "same_classes": same_classes,
                      "class_counts": {s: c.to_dict() for s, c in counts.items()},
                      "image_sizes_sampled": {f"{w}x{h}": n for (w, h), n in sizes.items()},
                      "unreadable_sampled": len(bad)})
print("wrote", args.results, "and", f"{args.out}/samples.png")
