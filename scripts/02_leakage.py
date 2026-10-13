"""Step 1b: how many validation images have a near-copy in train? (original dataset split)

Reports both raw (only L2-normalized) and mean-centered features. The threshold is derived
from the similarity distribution by `pick_threshold`, and a table of fixed thresholds is
reported next to it. Output: results JSON, a feature cache for step 1c, pair galleries.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from agriscan.data import find_dataset_root, find_splits, list_images
from agriscan.leakage import embed, make_embedder, pick_threshold, prepare, top1
from agriscan.utils import save_results, set_seed

GRID = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
BINS = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 1.01)]

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="auto", help="dataset root, or 'auto' to search /kaggle/input")
ap.add_argument("--out", default="artifacts/leakage")
ap.add_argument("--results", default="results/step1_leakage.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="use only N random images per split (CPU smoke test)")
ap.add_argument("--workers", type=int, default=2, help="DataLoader processes; use 0 on Windows (no fork)")
ap.add_argument("--no-pretrained", action="store_true", help="random weights, offline smoke test only")
args = ap.parse_args()

set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
root = find_dataset_root() if args.root == "auto" else Path(args.root)
print("dataset root:", root)
out = Path(args.out)
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

# The mean uses train+val only, so the 33 test images do not influence it.
_, mean = prepare(torch.cat([raw["train"], raw["valid"]]))
variants = {"centered": {s: prepare(x, mean)[0] for s, x in raw.items()},
            "raw": {s: F.normalize(x.float(), dim=1) for s, x in raw.items()}}

classes = sorted(tr.label.unique())
lab = {s: df[s].label.map({c: i for i, c in enumerate(classes)}).fillna(-1).to_numpy() for s in df}


def summarize(sims, thresholds):
    return {"n": len(sims), "quantiles": {str(q): float(np.quantile(sims, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "frac_at_least": {f"{t:g}": float((sims >= t).mean()) for t in thresholds}}


metrics, gallery = {}, {}
for vname, f in variants.items():
    v_sim, v_idx = top1(f["valid"], f["train"], device)
    other, _ = top1(f["valid"], f["train"], device, q_labels=lab["valid"], ref_labels=lab["train"], other_label=True)
    thr = pick_threshold(v_sim, other)
    # Thresholds read off the distribution of known non-copies (best match to a different class).
    qthr = {f"other_class_p{q * 100:g}": round(float(np.quantile(other, q)), 3) for q in (0.99, 0.999, 0.9999)}
    ths = sorted(set(GRID + [round(thr["final"], 3)] + list(qthr.values())))
    m = {"derived_threshold": thr, "control_quantile_thresholds": qthr,
         "valid_to_train": summarize(v_sim, ths),
         "valid_to_train_same_label_of_best_match": float((lab["train"][v_idx] == lab["valid"]).mean()),
         # Baseline: best match among OTHER classes (known non-copies).
         "control_valid_to_train_other_class": summarize(other, ths),
         # Duplication inside train itself (hides each image's match with itself).
         "control_train_to_train": summarize(top1(f["train"], f["train"], device, self_match=True)[0], ths)}
    if te is not None and len(te):
        m["control_test_to_train"] = summarize(top1(f["test"], f["train"], device)[0], ths)  # un-augmented photos
    metrics[vname] = m
    gallery[vname] = (v_sim, v_idx)
    print(f"\n===== {vname} features | derived threshold: {thr} | control quantile thresholds: {qthr}")
    for k, r in m.items():
        if isinstance(r, dict) and "n" in r:
            print(f"{k}: n={r['n']}\n  quantiles: { {a: round(b, 3) for a, b in r['quantiles'].items()} }"
                  f"\n  fraction >= threshold: { {a: round(b, 4) for a, b in r['frac_at_least'].items()} }")

# Galleries (centered features): top row = validation image, bottom row = its nearest train image.
v_sim, v_idx = gallery["centered"]
rng = np.random.default_rng(args.seed)
final = metrics["centered"]["derived_threshold"]["final"]
for lo, hi in sorted(set(BINS + [(round(final - 0.03, 3), round(final + 0.03, 3))])):
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
    fig.suptitle(f"centered: val (top) vs nearest train (bottom), similarity in [{lo}, {hi})  [{len(idx)} pairs in bin]")
    fig.tight_layout()
    fig.savefig(out / f"pairs_{lo:.3f}_{hi:.3f}.png", dpi=90)
    plt.close(fig)

save_results(args.results, name="step1_leakage", seed=args.seed,
             config={"root": str(root), "limit": args.limit, "pretrained": not args.no_pretrained,
                     "embedder": "mobilenet_v2 imagenet pooled 1280-d; 'centered' = minus train+val mean, then L2; 'raw' = L2 only",
                     "grid": GRID, "n_train": len(tr), "n_valid": len(va)},
             metrics=metrics)
print("\nwrote", args.results, "| galleries in", out)
