"""EXTRA: evidence for the grouping key. Checks the naming of the ORIGINAL PlantVillage repository (via the GitHub API at
run time) and of the Kaggle copy (from splits/clean_split.csv.gz), and compares grouping keys on the original split.
"""
import argparse
import gzip
import json
import time
import urllib.request

import pandas as pd

from agriscan.crosscheck import original_repo_naming, original_split
from agriscan.grouping import annotate, union_groups
from agriscan.utils import save_results

API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset"
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="splits/clean_split.csv.gz")
ap.add_argument("--results", default="results/extra_naming_check.json")
args = ap.parse_args()


def get_json(url, tries=4):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "agriscan-reimpl"})
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read()
                raw = gzip.decompress(raw) if r.headers.get("Content-Encoding") == "gzip" else raw
            return json.loads(raw)        # a truncated download fails here and is retried
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3)
    raise RuntimeError(f"GitHub API call failed: {url}: {last}")


color_sha = next(x["sha"] for x in get_json(f"{API}/contents/raw") if x["name"] == "color")
tree = get_json(f"{API}/git/trees/{color_sha}?recursive=1")
paths = [e["path"] for e in tree["tree"] if e["type"] == "blob" and "/" in e["path"]]
orig = original_repo_naming(paths)
orig["github_tree_truncated"] = bool(tree.get("truncated"))
orig["note"] = "GitHub truncates very large trees: the counts cover the returned part only (see n_files / n_classes)"

ann = annotate(pd.read_csv(args.csv)[["rel", "label"]])
ann["orig"] = ann.rel.map(original_split)
ann["uuid_group"] = ann.uuid.fillna(pd.Series("none" + ann.index.astype(str), index=ann.index))
ann["union_group"] = union_groups(ann, ["uuid", "class_base"])


def shared_with_train(col):
    tr = set(ann.loc[ann.orig == "train", col])
    va = ann[ann.orig == "val"]
    return float(va[col].isin(tr).mean())


sizes = ann.union_group.value_counts()
kaggle = {"n_files": int(len(ann)), "files_without_uuid_by_class": ann[ann.uuid.isna()].label.value_counts().to_dict(),
          "distinct_uuids": int(ann.uuid.nunique()), "distinct_class_base_names": int(ann.class_base.nunique()),
          "uuids_with_more_than_one_label": int((ann.groupby("uuid").label.nunique() > 1).sum()),
          "uuids_with_more_than_one_base_name": int((ann.groupby("uuid").base.nunique() > 1).sum()),
          "union_groups": int(len(sizes)), "union_largest_group": int(sizes.max()), "union_groups_over_40_images": int((sizes > 40).sum()),
          "original_split_val_images_with_a_group_member_in_train": {
              "uuid_only (files without uuid kept alone)": shared_with_train("uuid_group"),
              "class_and_base_name": shared_with_train("class_base"), "union_of_both": shared_with_train("union_group")}}
save_results(args.results, name="extra_naming_check", seed=0, config={"csv": args.csv, "api": API},
             metrics={"label": "EXTRA (naming evidence for the grouping key)", "original_repo": orig, "kaggle_copy": kaggle})
print(json.dumps({"original_repo": orig, "kaggle_copy": kaggle}, indent=2))
print("wrote", args.results)
