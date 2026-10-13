"""Step 1c: build a leakage-free split by keeping near-duplicate groups on one side.

Run after 02_leakage.py (it reuses the cached features) and after you picked a threshold
by looking at the pair galleries.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agriscan.leakage import group_split, near_dup_groups, prepare, top1
from agriscan.utils import save_results, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default="artifacts/leakage/feats.npz")
ap.add_argument("--threshold", type=float, required=True, help="cosine similarity at/above which two images are 'the same'")
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default=None, help="default: splits/clean_split_thr<threshold>.csv.gz")
ap.add_argument("--results", default=None)
args = ap.parse_args()

set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
tag = f"thr{args.threshold:.2f}"
csv_path = args.csv or f"splits/clean_split_{tag}.csv.gz"
res_path = args.results or f"results/step1_clean_split_{tag}.json"

z = np.load(args.feats, allow_pickle=True)
raw = torch.cat([torch.from_numpy(z["train"]), torch.from_numpy(z["valid"])])
rel = np.concatenate([z["rel_train"], z["rel_valid"]])
label = np.concatenate([z["label_train"], z["label_valid"]])
orig_val = np.arange(len(rel)) >= len(z["train"])  # the dataset's own split
feats, _ = prepare(raw)
y = pd.factorize(label, sort=True)[0]

# 1. Leakage of the original split at this threshold (same number as step 1b).
orig_sim, _ = top1(feats[orig_val], feats[~orig_val], device)

# 2. Group near-copies (over train+valid together), then re-split by group.
groups, n_cross = near_dup_groups(feats, y, args.threshold, device)
sizes = np.bincount(groups)
is_val = group_split(y, groups, args.val_frac, args.seed)

# 3. Verify: exact nearest neighbour of every clean-val image among clean-train images.
clean_sim, clean_idx = top1(feats[is_val], feats[~is_val], device)
same_label = y[~is_val][clean_idx] == y[is_val]
frac_per_class = pd.Series(is_val).groupby(y).mean()

m = {"threshold": args.threshold,
     "n_images": int(len(y)), "n_groups": int(len(sizes)),
     "group_size": {"max": int(sizes.max()), "mean": float(sizes.mean()),
                    "singletons": int((sizes == 1).sum()), "p99": float(np.quantile(sizes, 0.99))},
     "cross_label_pairs_above_threshold": n_cross,
     "original_split": {"n_val": int(orig_val.sum()), "val_with_near_copy_in_train": int((orig_sim >= args.threshold).sum()),
                        "fraction": float((orig_sim >= args.threshold).mean())},
     "clean_split": {"n_train": int((~is_val).sum()), "n_val": int(is_val.sum()),
                     "val_fraction_overall": float(is_val.mean()),
                     "val_fraction_per_class_min_max": [float(frac_per_class.min()), float(frac_per_class.max())],
                     "val_with_near_copy_in_train": int((clean_sim >= args.threshold).sum()),
                     "of_which_same_label": int(((clean_sim >= args.threshold) & same_label).sum())}}
for k, v in m.items():
    print(f"{k}: {v}")

Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"rel": rel, "label": label, "split": np.where(is_val, "val", "train"), "group": groups}) \
    .to_csv(csv_path, index=False)
save_results(res_path, name="step1_clean_split", seed=args.seed,
             config={"feats": args.feats, "val_frac": args.val_frac, "threshold": args.threshold}, metrics=m)
print("wrote", csv_path, "and", res_path)
