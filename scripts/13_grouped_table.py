"""Tables for the second protocol (photo- and leaf-grouped splits), generated from results/*.json (no hand-typed numbers).

Writes results/RESULTS_GROUPED.md and, with --readme, replaces the block between <!-- GROUPED:START --> and
<!-- GROUPED:END -->. Runs that do not exist yet appear as 'not run yet'.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/RESULTS_GROUPED.md")
ap.add_argument("--readme", default=None)
args = ap.parse_args()
R = Path(args.results)


def load(rel):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def run(arch, split, seed=0):
    return load(f"runs/{arch}__{split}__s{seed}__base.json")


pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
f4 = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
pp = lambda x: f"{100 * x:+.2f} pp"  # noqa: E731
sp, nm, xc = load("step8_grouped_splits.json"), load("extra_naming_check.json"), load("extra_filename_crosscheck.json")
L = []

# ---- the failed first attempt -------------------------------------------------------------------------------
if xc:
    o, c = xc["original_split"], xc["clean_split_used_in_this_repo"]
    L += ["### First attempt: similarity-based split (kept, and reported as failed)", "",
          "The first 'clean' split grouped images by embedding similarity. It did **not** remove leakage: by the file-name check "
          f"{pct(o['fraction_of_all_val'])} of the original validation images have a same-source image in train, and still "
          f"{pct(c['fraction_of_all_val'])} of that split's validation images do. Its results stay in the tables above.", ""]

# ---- construction and checks --------------------------------------------------------------------------------
if sp:
    m = sp["metrics"]
    cov, nmg = m["leaf_coverage"], m["naming"]
    L += ["### Second protocol: grouped splits (construction and checks)", "",
          f"Images are linked into groups (union of: same UUID, same class + base file name, and for the leaf split also the "
          f"authors' leaf id). Whole groups go to train / validation / test (target {int(100 * m['fracs_train_val_test'][0])}/"
          f"{int(100 * m['fracs_train_val_test'][1])}/{int(100 * m['fracs_train_val_test'][2])}, stratified by class, seed {sp['seed']}). "
          f"Input: {m['n_files']} files. Split fingerprints are committed in `splits/split_hashes.json` and re-checked on Kaggle.", "",
          "| check | photo-grouped | leaf+photo-grouped (primary) |", "|---|---|---|"]
    ph, lf = m["photo"], m["leaf"]
    row = lambda label, f: L.append(f"| {label} | {f(ph)} | {f(lf)} |")  # noqa: E731
    row("images train / val / test", lambda c: " / ".join(str(c["n"][p]) for p in ("train", "val", "test")))
    row("groups (largest)", lambda c: f"{c['n_groups']} ({c['largest_group']})")
    row("groups over 100 images", lambda c: str(c["groups_over_100_images"]))
    row("per-class val share (min-max)", lambda c: f"{c['per_class_share_min_max']['val'][0]:.3f}-{c['per_class_share_min_max']['val'][1]:.3f}")
    row("per-class test share (min-max)", lambda c: f"{c['per_class_share_min_max']['test'][0]:.3f}-{c['per_class_share_min_max']['test'][1]:.3f}")
    row("groups / UUIDs / base names shared across parts", lambda c: f"{c['groups_straddling_parts']} / {c['uuids_straddling_parts']} / {c['class_base_names_straddling_parts']}")
    row("leaf ids shared across parts", lambda c: str(c["leaf_ids_straddling_parts"]))
    row("test images with a leaf id", lambda c: str(c["test_files_with_leaf_id"]))
    for part in ("val", "test"):
        row(f"{part} images (with a leaf id) whose leaf is also in train", lambda c, part=part: f"{pct(c['same_leaf_leakage_of_mapped_files'][part]['share_whose_leaf_is_in_train'])} of {c['same_leaf_leakage_of_mapped_files'][part]['n_with_leaf_id']}")
    L += ["", f"**Leaf-map coverage: {pct(cov['fraction'])} of the files** ({cov['files_with_leaf_id']} of {m['n_files']}) have a leaf id from the authors' map; "
          f"the other {pct(1 - cov['fraction'])} are grouped by photo only, so same-leaf leakage can remain for them even in the leaf split. "
          f"Per-class coverage ranges {pct(cov['per_class_min_max'][0])}-{pct(cov['per_class_min_max'][1])}; classes with zero coverage: "
          f"{', '.join(cov['classes_with_zero_coverage']) or 'none'}. The map is downloaded at run time (sha256 `{m['leaf_map']['sha256'][:12]}...`) and is not committed.",
          f"Files without a UUID: {nmg['files_without_uuid']} ({', '.join(f'{k}: {v}' for k, v in nmg['classes_without_uuid'].items()) or 'none'}); "
          f"files named with ' copy': {nmg['copy_named_files']}.", ""]

# ---- naming evidence ----------------------------------------------------------------------------------------
if nm:
    o, k = nm["metrics"]["original_repo"], nm["metrics"]["kaggle_copy"]
    sib = k["original_split_val_images_with_a_group_member_in_train"]
    L += ["### Evidence for the grouping key (naming check)", "",
          f"Original PlantVillage repository (GitHub API, {o['n_files']} files in {o['n_classes']} classes; tree truncated by GitHub: {o['github_tree_truncated']}): "
          f"{o['n_with_uuid_prefix']} files have a UUID prefix and {o['n_distinct_uuids']} UUIDs are distinct ({o['uuids_used_by_more_than_one_file']} UUIDs are used by more than one file); "
          f"{sum(o['files_without_uuid_by_class'].values())} files have no UUID ({', '.join(f'{c}: {n}' for c, n in o['files_without_uuid_by_class'].items()) or 'none'}); "
          f"{o['files_with_augmentation_suffix']} files carry an augmentation suffix; {o['copy_named_files']} files are named ' copy', and in "
          f"{o['of_which_with_more_than_one_uuid']} of {o['groups_with_a_copy_file']} (class, base name) groups that contain a copy the files have more than one UUID.", "",
          f"Kaggle copy: {sum(k['files_without_uuid_by_class'].values())} files have no UUID ({', '.join(f'{c}: {n}' for c, n in k['files_without_uuid_by_class'].items())}); "
          f"{k['uuids_with_more_than_one_label']} UUIDs appear under more than one label and {k['uuids_with_more_than_one_base_name']} under more than one base name. "
          f"On the dataset's original split, the share of validation images with a group member in train is "
          f"{', '.join(f'{pct(v)} ({n})' for n, v in sib.items())}. The union key gives {k['union_groups']} groups, the largest with {k['union_largest_group']} images "
          f"({k['union_groups_over_40_images']} groups over 40).", ""]

# ---- results ------------------------------------------------------------------------------------------------
L += ["### Results by protocol (one seed per grouped run)", "",
      "Final-epoch metrics. Validation accuracy for 'clean' is the mean of 3 seeds; every other row is one run. "
      "**Headline number: the leaf-mapped test accuracy on the leaf-grouped split** (files covered by the authors' leaf map; same-leaf leakage is 0.0% there).", "",
      "| backbone | protocol | val acc | test acc (all) | test acc, leaf-mapped files (n) | test acc, files without leaf id (n) |", "|---|---|---|---|---|---|"]
head = {}
for arch in ("mobilenet_v2", "efficientnet_b3"):
    o = run(arch, "original")
    cl = [r for r in (run(arch, "clean", s) for s in (0, 1, 2)) if r]
    rows = [("original split (dataset's own)", f4(o["metrics"]["val_acc"]) if o else "not run yet", "-", "-", "-"),
            ("similarity 'clean' split (failed attempt, 3 seeds)", f4(np.mean([r["metrics"]["val_acc"] for r in cl])) if cl else "not run yet", "-", "-", "-")]
    for split, label in (("photo", "photo-grouped"), ("leaf", "leaf+photo-grouped (primary)")):
        r = run(arch, split)
        if not r:
            rows.append((label, "not run yet", "not run yet", "not run yet", "not run yet"))
            continue
        t = r["metrics"]["test"]
        rows.append((label, f4(r["metrics"]["val_acc"]), f4(t["acc"]), f"{f4(t['leaf_mapped']['acc'])} ({t['leaf_mapped']['n']})",
                     f"{f4(t['not_leaf_mapped']['acc'])} ({t['not_leaf_mapped']['n']})"))
        if split == "leaf":
            head[arch] = (t["leaf_mapped"]["acc"], t["leaf_mapped"]["n"])
    for label, *cells in rows:
        L.append(f"| {arch} | {label} | " + " | ".join(cells) + " |")
L += [""]
if head:
    L += ["**Headline (leaf-mapped test accuracy, leaf-grouped split, one seed): " +
          "; ".join(f"{a} {pct(v)} (n={n})" for a, (v, n) in head.items()) + ".** It describes lab-condition images of leaves the model has not seen "
          "in training, for the part of the data covered by the authors' leaf map; it says nothing about field photos.", ""]
else:
    L += ["The grouped runs have not been executed yet, so there is no headline number.", ""]

# ---- observations -------------------------------------------------------------------------------------------
obs = []
for arch in ("mobilenet_v2", "efficientnet_b3"):
    o, p, lf = run(arch, "original"), run(arch, "photo"), run(arch, "leaf")
    if o and p:
        obs.append(f"{arch}: original-split val {f4(o['metrics']['val_acc'])} -> photo-grouped test {f4(p['metrics']['test']['acc'])} "
                   f"({pp(p['metrics']['test']['acc'] - o['metrics']['val_acc'])}; different images and a held-out part, one seed).")
    if p and lf:
        pm, lm = p["metrics"]["test"]["leaf_mapped"]["acc"], lf["metrics"]["test"]["leaf_mapped"]["acc"]
        obs.append(f"{arch}: leaf-mapped test accuracy {f4(pm)} on the photo-grouped split vs {f4(lm)} on the leaf-grouped split ({pp(lm - pm)}); "
                   f"on the photo-grouped split these images mostly share a leaf with training images, on the leaf-grouped split none do.")
L += ["### Observations (generated from the numbers above)", ""] + ([f"- {x}" for x in obs] if obs else ["- No grouped runs yet."]) + [""]

text = "\n".join(L)
Path(args.out).write_text(text, encoding="utf-8")
print(text)
if args.readme:
    p = Path(args.readme)
    t = p.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- GROUPED:START -->).*?(<!-- GROUPED:END -->)", re.S)
    assert pat.search(t), "README has no GROUPED markers"
    p.write_text(pat.sub(lambda m: f"{m.group(1)}\n_Generated by `scripts/13_grouped_table.py` from `results/`; do not edit by hand._\n\n{text}\n{m.group(2)}", t), encoding="utf-8")
    print("updated", args.readme)
