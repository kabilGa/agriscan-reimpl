"""Step 1c: build a leakage-free split by keeping near-duplicate groups on one side.

Run after 02_leakage.py (it reuses the cached features). `--threshold auto` starts from the value
derived there from the similarity distribution; if that gives a degenerate split (chained giant
groups, or a class whose validation share is far from the target) it tries stricter thresholds.
Every attempt is recorded in the results JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agriscan.leakage import group_split, near_dup_groups, prepare, top1
from agriscan.utils import save_results, set_seed

LADDER = [0.90, 0.95, 0.98]  # stricter fallbacks, tried in order if the derived threshold is degenerate

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default="artifacts/leakage/feats.npz")
ap.add_argument("--threshold", default="auto", help="cosine similarity at/above which two images are 'the same', or 'auto'")
ap.add_argument("--leakage-results", default="results/step1_leakage.json")
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default="splits/clean_split.csv.gz")
ap.add_argument("--results", default="results/step1_clean_split.json")
args = ap.parse_args()

set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
if args.threshold == "auto":
    derived = json.loads(Path(args.leakage_results).read_text())["metrics"]["centered"]["derived_threshold"]["final"]
    candidates = [derived] + [t for t in LADDER if t > derived]
    print(f"threshold = auto -> derived {derived:.4f}, fallbacks {candidates[1:]}")
else:
    candidates = [float(args.threshold)]

z = np.load(args.feats, allow_pickle=True)
raw = torch.cat([torch.from_numpy(z["train"]), torch.from_numpy(z["valid"])])
rel = np.concatenate([z["rel_train"], z["rel_valid"]])
label = np.concatenate([z["label_train"], z["label_valid"]])
orig_val = np.arange(len(rel)) >= len(z["train"])  # the dataset's own split
feats, _ = prepare(raw)
y = pd.factorize(label, sort=True)[0]
orig_sim, _ = top1(feats[orig_val], feats[~orig_val], device)   # val -> train best match, original split


def attempt(t):
    try:
        groups, n_cross = near_dup_groups(feats, y, t, device)
    except RuntimeError as e:  # too many near pairs: the threshold is far too low; treat as a failed attempt
        return None, None, {"threshold": t, "ok": False, "reasons": [str(e)]}
    sizes = np.bincount(groups)
    is_val = group_split(y, groups, args.val_frac, args.seed)
    per_class = pd.Series(is_val).groupby(y).mean()
    reasons = []
    if sizes.max() > max(0.01 * len(y), 50):
        reasons.append(f"giant group ({sizes.max()} images)")
    if per_class.min() < 0.5 * args.val_frac or per_class.max() > 1.5 * args.val_frac:
        reasons.append(f"class val share {per_class.min():.3f}-{per_class.max():.3f} vs target {args.val_frac}")
    info = {"threshold": t, "ok": not reasons, "reasons": reasons, "n_groups": int(len(sizes)),
            "max_group": int(sizes.max()), "mean_group": float(sizes.mean()), "singletons": int((sizes == 1).sum()),
            "p99_group": float(np.quantile(sizes, 0.99)), "cross_label_pairs": n_cross,
            "val_share_per_class_min_max": [float(per_class.min()), float(per_class.max())],
            "original_split_val_with_near_copy_fraction": float((orig_sim >= t).mean())}
    return groups, is_val, info


attempts = []
for thr in candidates:  # stops at the first healthy split; otherwise ends on the last (strictest) attempt
    groups, is_val, info = attempt(thr)
    attempts.append(info)
    print(f"attempt thr={thr:.4f}: ok={info['ok']} {info['reasons']} max_group={info.get('max_group')}")
    if info["ok"]:
        break
healthy = attempts[-1]["ok"]
if groups is None:
    raise SystemExit("Every threshold produced too many near-duplicate pairs; cannot build a split (see attempts above).")
if not healthy:  # keep going (an unattended run must not crash) but flag it loudly in the JSON
    print("WARNING: no threshold gave a healthy split; using the strictest attempt, flagged in the JSON")

# Verify: exact nearest neighbour of every clean-val image among clean-train images.
clean_sim, clean_idx = top1(feats[is_val], feats[~is_val], device)
same_label = y[~is_val][clean_idx] == y[is_val]
frac_per_class = pd.Series(is_val).groupby(y).mean()

sens = [attempt(t)[2] for t in sorted({round(max(thr - 0.05, 0.0), 4), round(min(thr + 0.03, 0.999), 4)} - {thr})]  # may contain failed attempts
source = "manual" if args.threshold != "auto" else ("derived" if thr == candidates[0] else "fallback_ladder")
m = {"threshold": thr, "threshold_source": source, "split_is_healthy": healthy,
     "attempts": attempts, "sensitivity": sens, "n_images": int(len(y)),
     "original_split": {"n_val": int(orig_val.sum()), "val_with_near_copy_in_train": int((orig_sim >= thr).sum()),
                        "fraction": float((orig_sim >= thr).mean())},
     "clean_split": {"n_train": int((~is_val).sum()), "n_val": int(is_val.sum()),
                     "val_fraction_overall": float(is_val.mean()),
                     "val_fraction_per_class_min_max": [float(frac_per_class.min()), float(frac_per_class.max())],
                     "val_with_near_copy_in_train": int((clean_sim >= thr).sum()),
                     "of_which_same_label": int(((clean_sim >= thr) & same_label).sum())}}
print(json.dumps(m, indent=2))

Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"rel": rel, "label": label, "split": np.where(is_val, "val", "train"), "group": groups}) \
    .to_csv(args.csv, index=False)
save_results(args.results, name="step1_clean_split", seed=args.seed,
             config={"feats": args.feats, "val_frac": args.val_frac, "threshold_arg": args.threshold, "csv": args.csv},
             metrics=m)
print("wrote", args.csv, "and", args.results)
