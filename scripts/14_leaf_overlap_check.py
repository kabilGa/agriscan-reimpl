"""EXTRA: independent recomputation of the same-leaf overlap between each split's validation/test images and its OWN training part.

Reads the split files and the authors' leaf map (downloaded into artifacts/ if missing; its hash must match the committed one).
For every leaf-mapped image of the part, counts the training images of the same leaf. Also compares the share with the value
stored by the split build (results/step8_grouped_splits.json), so two separate computations must agree.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from agriscan.grouping import annotate, load_leaf_map
from agriscan.utils import save_results

ap = argparse.ArgumentParser()
ap.add_argument("--splits-dir", default="splits")
ap.add_argument("--leaf-map", default="artifacts/leaf-map.json")
ap.add_argument("--expected", default="splits/split_hashes.json")
ap.add_argument("--stored", default="results/step8_grouped_splits.json", help="split-build JSON to compare the shares with")
ap.add_argument("--results", default="results/extra_leaf_overlap_check.json")
args = ap.parse_args()

leaf_map, sha, _ = load_leaf_map(args.leaf_map)
exp = Path(args.expected)
if exp.exists() and json.loads(exp.read_text())["leaf_map_sha256"] != sha:
    raise SystemExit("The leaf map differs from the one the splits were built with; refusing to compare.")
stored = json.loads(Path(args.stored).read_text())["metrics"] if Path(args.stored).exists() else None

out = {"label": "EXTRA (independent recomputation of same-leaf overlap)", "leaf_map_sha256": sha}
for name in ("photo", "leaf"):
    sp = pd.read_csv(Path(args.splits_dir) / f"{name}_split.csv.gz")
    ann = annotate(sp[["rel", "label"]], leaf_map).merge(sp[["rel", "split"]], on="rel")
    mapped = ann[ann.leaf.notna()]
    train_counts = mapped[mapped.split == "train"].leaf.value_counts()
    out[name] = {}
    for part in ("val", "test"):
        t = mapped[mapped.split == part]
        k = t.leaf.map(train_counts).fillna(0)
        share = float((k >= 1).mean())
        rec = {"n_leaf_mapped": int(len(t)), "n_with_same_leaf_in_train": int((k >= 1).sum()), "share_with_same_leaf_in_train": share,
               "same_leaf_train_images_per_image": {"mean": float(k.mean()), "median": float(k.median()), "min": int(k.min()), "max": int(k.max())},
               "distinct_leaves": int(t.leaf.nunique()), "distinct_leaves_also_in_train": int(t.drop_duplicates("leaf").leaf.isin(train_counts.index).sum())}
        if stored:
            ref = stored[name]["same_leaf_leakage_of_mapped_files"][part]
            rec["agrees_with_split_build"] = bool(ref["n_with_leaf_id"] == len(t) and abs(ref["share_whose_leaf_is_in_train"] - share) < 1e-9)
        out[name][part] = rec
        print(f"{name:5s} {part:4s}: {rec['n_with_same_leaf_in_train']} of {rec['n_leaf_mapped']} leaf-mapped images ({100 * share:.2f}%) have >=1 same-leaf training image; "
              f"mean {k.mean():.1f} median {k.median():.0f} per image" + (f" | agrees with split build: {rec['agrees_with_split_build']}" if stored else ""))
save_results(args.results, name="extra_leaf_overlap_check", seed=0, config={"splits_dir": args.splits_dir}, metrics=out)
print("wrote", args.results)
