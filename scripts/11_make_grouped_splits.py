"""Build the photo-grouped and leaf+photo-grouped 70/15/15 splits (second protocol).

Input is a list of (rel, label): either splits/clean_split.csv.gz (laptop, no dataset needed) or the dataset folder
itself (Kaggle). The authors' leaf map is downloaded at run time into artifacts/ (git-ignored) and never committed.
The result is fingerprinted and compared with splits/split_hashes.json, so a Kaggle rebuild is provably the same split.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from agriscan.data import find_dataset_root, find_splits, list_images
from agriscan.grouping import (FRACS, LEAF_MAP_URL, build_grouped_splits, files_hash, load_leaf_map, split_checks,
                               split_hash)
from agriscan.utils import save_results

ap = argparse.ArgumentParser()
ap.add_argument("--files", default="splits/clean_split.csv.gz", help="CSV with columns rel,label (ignored if --root is given)")
ap.add_argument("--root", default=None, help="dataset root or 'auto' (list the files from the dataset folder, as on Kaggle)")
ap.add_argument("--leaf-map", default="artifacts/leaf-map.json", help="cache path; downloaded here if missing")
ap.add_argument("--out", default="splits")
ap.add_argument("--results", default="results/step8_grouped_splits.json")
ap.add_argument("--expected", default="splits/split_hashes.json")
ap.add_argument("--write-expected", action="store_true", help="record these hashes as the expected ones (laptop, once)")
ap.add_argument("--allow-mismatch", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

if args.root:
    root = find_dataset_root() if args.root == "auto" else Path(args.root)
    dirs = find_splits(root)
    files = pd.concat([list_images(dirs[s], s, root) for s in ("train", "valid")])[["rel", "label"]]
    source = f"dataset folder {root}"
else:
    files = pd.read_csv(args.files)[["rel", "label"]]
    source = f"file list {args.files}"
files = files.sort_values("rel").reset_index(drop=True)
print(f"{len(files)} files from {source}")

leaf_map, leaf_sha, downloaded = load_leaf_map(args.leaf_map)
print(f"leaf map: {len(leaf_map)} keys, sha256 {leaf_sha[:16]}..., downloaded now: {downloaded}")
ann, splits = build_grouped_splits(files, leaf_map, args.seed)
hashes = {"files": files_hash(files), "leaf_map_sha256": leaf_sha, "photo": split_hash(splits["photo"]), "leaf": split_hash(splits["leaf"])}

exp_path = Path(args.expected)
if args.write_expected:
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    exp_path.write_text(json.dumps({**hashes, "n_files": len(files), "seed": args.seed, "fracs": FRACS}, indent=2))
    print("wrote expected hashes to", exp_path)
elif exp_path.exists():
    exp = json.loads(exp_path.read_text())
    bad = {k: (exp.get(k), hashes[k]) for k in hashes if exp.get(k) != hashes[k]}
    for k in hashes:
        print(f"  hash {k:16s} {'MATCH' if k not in bad else 'MISMATCH'}")
    if bad and not args.allow_mismatch:
        print("The rebuilt splits differ from the committed ones (different file list or leaf map). Stopping; "
              "use --allow-mismatch only if you accept that the results will not match the committed split files.", file=sys.stderr)
        for k, (a, b) in bad.items():
            print(f"   {k}: expected {a}\n   {' ' * len(k)}  got      {b}", file=sys.stderr)
        sys.exit(2)
else:
    print("no expected hashes file found; skipping the hash check")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for name, df in splits.items():
    # rel, label, split, group, leaf_mapped (no leaf ids: they come from the leaf map). mtime=0 keeps the bytes reproducible.
    df.to_csv(out / f"{name}_split.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

cov = ann.groupby("label").leaf_mapped.mean()
no_uuid = ann[ann.uuid.isna()]
metrics = {"fracs_train_val_test": FRACS, "n_files": len(files), "input": source, "hashes": hashes,
           "leaf_map": {"url": LEAF_MAP_URL, "sha256": leaf_sha, "n_keys": len(leaf_map), "downloaded_this_run": downloaded,
                        "note": "downloaded at run time, never committed (the original repository states no licence)"},
           "leaf_coverage": {"files_with_leaf_id": int(ann.leaf_mapped.sum()), "fraction": float(ann.leaf_mapped.mean()),
                             "per_class_min_max": [float(cov.min()), float(cov.max())],
                             "classes_with_zero_coverage": sorted(cov[cov == 0].index.tolist())},
           "naming": {"files_without_uuid": int(len(no_uuid)), "classes_without_uuid": no_uuid.label.value_counts().to_dict(),
                      "distinct_uuids": int(ann.uuid.nunique()), "distinct_class_base_names": int(ann.class_base.nunique()),
                      "copy_named_files": int(ann.rel.str.contains(r"\scopy", regex=True).sum())},
           "photo": split_checks(ann, splits["photo"]), "leaf": split_checks(ann, splits["leaf"])}
save_results(args.results, name="step8_grouped_splits", seed=args.seed,
             config={"input": source, "leaf_map": args.leaf_map, "grouping": {"photo": "uuid OR class+base name", "leaf": "uuid OR class+base name OR authors' leaf id"}},
             metrics=metrics)
for name in ("photo", "leaf"):
    c = metrics[name]
    print(f"\n{name}: n={c['n']} groups={c['n_groups']} largest={c['largest_group']} straddling groups={c['groups_straddling_parts']}")
    print("   per-class share min/max:", {k: [round(x, 3) for x in v] for k, v in c["per_class_share_min_max"].items()})
    print("   same-leaf leakage of leaf-mapped files:", c["same_leaf_leakage_of_mapped_files"])
print(f"\nleaf-map coverage: {metrics['leaf_coverage']['fraction']:.1%} of files | files without uuid: {metrics['naming']['files_without_uuid']} {metrics['naming']['classes_without_uuid']}")
print("wrote", out / "photo_split.csv.gz", out / "leaf_split.csv.gz", "and", args.results)
