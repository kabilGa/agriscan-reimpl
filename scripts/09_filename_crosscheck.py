"""EXTRA: cross-check the leakage estimate and the clean split against filename-derived source ids.

Needs only splits/clean_split.csv.gz (paths, labels, split, group): no images, no GPU. See agriscan.crosscheck
for the assumption this relies on.
"""
import argparse

import pandas as pd

from agriscan.crosscheck import group_capture, prepare, sibling_leakage
from agriscan.utils import save_results

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="splits/clean_split.csv.gz")
ap.add_argument("--results", default="results/extra_filename_crosscheck.json")
args = ap.parse_args()

df = prepare(pd.read_csv(args.csv))
m = {"label": "EXTRA (filename-derived cross-check)",
     "assumption": "files sharing the UUID before '___' in their name are copies of one source photo (not verified against the dataset's documentation)",
     "n_images": int(len(df)), "n_images_with_key": int(df.key.notna().sum()),
     "original_split": sibling_leakage(df, "orig"),
     "clean_split_used_in_this_repo": sibling_leakage(df, "split"),
     "embedding_groups": group_capture(df)}
for name in ("original_split", "clean_split_used_in_this_repo"):
    r = m[name]
    print(f"{name}: {r['with_sibling_in_train']} of {r['n_val']} validation images ({r['fraction_of_all_val']:.1%}; "
          f"{r['fraction_of_keyed_val']:.1%} of those with a parsable id) have a same-source image in train")
g = m["embedding_groups"]
print(f"embedding groups kept {g['sources_kept_in_one_group']} of {g['sources_with_more_than_one_image']} multi-image sources in one group; "
      f"{g['images_grouped_with_a_sibling']} of {g['images_with_a_sibling']} images share a group with a sibling")
save_results(args.results, name="extra_filename_crosscheck", seed=0, config={"csv": args.csv}, metrics=m)
print("wrote", args.results)
