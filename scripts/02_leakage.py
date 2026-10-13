"""Step 1b: how many validation images have a near-copy in train? (original dataset split)

Output: results JSON with the numbers, a feature cache for step 1c, and pair galleries
to eyeball what 'similarity 0.95' actually looks like.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from agriscan.data import find_splits, list_images
from agriscan.leakage import embed, make_embedder, prepare, top1
from agriscan.utils import save_results, set_seed

THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
BINS = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 1.01)]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="artifacts/leakage")
ap.add_argument("--results", default="results/step1_leakage.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="use only N random images per split (CPU smoke test)")
ap.add_argument("--workers", type=int, default=2, help="DataLoader processes; use 0 on Windows (no fork)")
ap.add_argument("--no-pretrained", action="store_true", help="random weights, offline smoke test only")
args = ap.parse_args()

set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
root, out = Path(args.root), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

dirs = find_splits(root)
df = {s: list_images(d, s, root) for s, d in dirs.items()}
if args.limit:
    df = {s: d.sample(min(args.limit, len(d)), random_state=args.seed).reset_index(drop=True) for s, d in df.items()}
tr, va = df["train"], df["valid"]
te = df.get("test")

# Separate cache name per --limit, so a smoke run can never be mistaken for the full run.
cache = out / (f"feats_limit{args.limit}.npz" if args.limit else "feats.npz")
if cache.exists():
    z = np.load(cache, allow_pickle=True)
    raw = {s: torch.from_numpy(z[s]) for s in ("train", "valid", "test") if s in z}
    print("loaded cached features from", cache)
else:
    model = make_embedder(pretrained=not args.no_pretrained)
    raw = {s: embed(d.path.tolist(), model, device, workers=args.workers) for s, d in df.items()}
    np.savez(cache, **{s: f.numpy() for s, f in raw.items()},
             **{f"rel_{s}": d.rel.to_numpy() for s, d in df.items()},
             **{f"label_{s}": d.label.to_numpy() for s, d in df.items()})

# The mean is computed on train+val only, so the 33 test images do not influence it.
_, mean = prepare(torch.cat([raw["train"], raw["valid"]]))
f = {s: prepare(x, mean)[0] for s, x in raw.items()}

classes = sorted(tr.label.unique())
lab = {s: df[s].label.map({c: i for i, c in enumerate(classes)}).fillna(-1).to_numpy() for s in df}


def summarize(sims):
    return {"n": len(sims), "quantiles": {str(q): float(np.quantile(sims, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "frac_at_least": {str(t): float((sims >= t).mean()) for t in THRESHOLDS}}


v_sim, v_idx = top1(f["valid"], f["train"], device)
res = {"valid_to_train": summarize(v_sim),
       "valid_to_train_same_label_of_best_match": float((lab["train"][v_idx] == lab["valid"]).mean()),
       # Baseline: best match among OTHER classes. High values here mean the threshold must be high.
       "control_valid_to_train_other_class": summarize(
           top1(f["valid"], f["train"], device, q_labels=lab["valid"], ref_labels=lab["train"], other_label=True)[0]),
       # Duplication inside train itself (hides each image's match with itself).
       "control_train_to_train": summarize(top1(f["train"], f["train"], device, self_match=True)[0])}
if te is not None and len(te):
    res["control_test_to_train"] = summarize(top1(f["test"], f["train"], device)[0])  # un-augmented photos

for name, r in res.items():
    if isinstance(r, dict):
        print(f"\n{name}: n={r['n']}")
        print("  quantiles:", {k: round(v, 3) for k, v in r["quantiles"].items()})
        print("  fraction >= threshold:", {k: round(v, 4) for k, v in r["frac_at_least"].items()})

# Galleries: top row = validation image, bottom row = its nearest train image.
rng = np.random.default_rng(args.seed)
for lo, hi in BINS:
    idx = np.flatnonzero((v_sim >= lo) & (v_sim < hi))
    if len(idx) == 0:
        continue
    pick = rng.choice(idx, size=min(6, len(idx)), replace=False)
    fig, axes = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 5), squeeze=False)
    for k, i in enumerate(pick):
        axes[0, k].imshow(Image.open(va.path[i]))
        axes[0, k].set_title(f"sim={v_sim[i]:.3f}", fontsize=8)
        axes[1, k].imshow(Image.open(tr.path[v_idx[i]]))
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle(f"val (top) vs nearest train (bottom), similarity in [{lo}, {hi})  [{len(idx)} pairs in bin]")
    fig.tight_layout()
    fig.savefig(out / f"pairs_{lo:.2f}_{hi:.2f}.png", dpi=90)
    plt.close(fig)

save_results(args.results, name="step1_leakage", seed=args.seed,
             config={"root": str(root), "limit": args.limit, "pretrained": not args.no_pretrained,
                     "embedder": "mobilenet_v2 imagenet pooled 1280-d, centered, L2-normalized",
                     "n_train": len(tr), "n_valid": len(va)},
             metrics=res)
print("\nwrote", args.results, "| galleries in", out)
