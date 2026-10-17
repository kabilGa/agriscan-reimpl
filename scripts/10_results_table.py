"""Build every results table of the README from the JSON files in results/ (no hand-typed numbers).

Writes results/RESULTS.md and, with --readme, replaces the block between <!-- RESULTS:START --> and
<!-- RESULTS:END --> in that file. The 'Observations' list is computed too, with neutral wording: it says what
a number is relative to a seed range, never why.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/RESULTS.md")
ap.add_argument("--readme", default=None)
args = ap.parse_args()

R = Path(args.results)
load = lambda p: json.loads((R / p).read_text())  # noqa: E731
# First run only (original and similarity-based 'clean' splits); the grouped-protocol runs are tabulated by scripts/13_grouped_table.py.
runs = [r for r in (json.loads(p.read_text()) for p in sorted((R / "runs").glob("*.json"))) if r["config"]["split"] in ("original", "clean")]
status, ins = load("run_status.json"), load("step1_inspect.json")["metrics"]
leak, split = load("step1_leakage.json")["metrics"], load("step1_clean_split.json")["metrics"]
yolo, bg = load("step5_yolo_eval.json")["metrics"], load("extra_background_check.json")["metrics"]
xc = load("extra_filename_crosscheck.json")["metrics"] if (R / "extra_filename_crosscheck.json").exists() else None

groups = defaultdict(list)
for r in runs:
    groups[(r["config"]["arch"], r["config"]["split"], r["config"]["tag"])].append(r)


def stats(arch, sp, tag="base"):
    g = groups[(arch, sp, tag)]
    a = np.array([x["metrics"]["val_acc"] for x in g])
    m = lambda k: float(np.mean([x["metrics"][k] for x in g]))  # noqa: E731
    return {"n": len(g), "mean": a.mean(), "min": a.min(), "max": a.max(), "recall": m("macro_recall"), "ece": m("ece_15bins"),
            "mb": g[0]["metrics"]["model_size_mb_fp32"], "ips": m("forward_img_per_s"), "dev": g[0]["metrics"]["speed_device"],
            "minutes": m("train_seconds") / 60, "img": g[0]["metrics"]["img_size"], "roles": sorted({x["config"]["role"] for x in g}),
            "seeds": sorted(x["seed"] for x in g)}


pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
f4 = lambda x: f"{x:.4f}"  # noqa: E731
pp = lambda x: f"{100 * x:+.2f} pp"  # noqa: E731
L = []
env = status["env"]
commits = sorted({r["env"]["git_commit"] for r in runs} | {env["git_commit"]})
L += ["### Run provenance", "",
      f"All {len(runs)} training runs and the step 1 / step 5 / extra stages come from one Kaggle session on {env['device']} "
      f"(torch {env['torch']}, torchvision {env['torchvision']}, ultralytics {env['ultralytics']}, Python {env['python']}), "
      f"wall time {status['elapsed_min'] / 60:.1f} h, code commit {', '.join(commits)}, stages: "
      f"{sum(s['state'] == 'done' for s in status['stages'])} of {len(status['stages'])} done. "
      f"Dataset: {ins['n_images']['train']} train + {ins['n_images']['valid']} valid images, {ins['n_classes']} classes "
      f"(per-class train {min(ins['class_counts']['train'].values())}-{max(ins['class_counts']['train'].values())}), "
      f"{ins['n_images']['test']} unlabeled test images used only as a control.", ""]

L += ["### Classifiers: original split vs clean split", "",
      "Final-epoch validation metrics (no best-epoch selection). 'Original' is the dataset's own train/valid split; "
      "'clean' is the embedding-grouped split built here (see the leakage sections: it is NOT free of leakage). "
      "Speed = forward-only images/s on one batch on the GPU above; time = minutes per run.", "",
      "| backbone | input px | split | seeds (n) | val acc, mean (min-max) | macro recall | ECE | size MB | fwd img/s | min/run | role |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for arch in ("mobilenet_v2", "efficientnet_b3", "resnet50"):
    for sp in ("clean", "original"):
        if (arch, sp, "base") in groups:
            s = stats(arch, sp)
            rng = f"{f4(s['min'])}-{f4(s['max'])}" if s["n"] > 1 else "single run"
            L.append(f"| {arch} | {s['img']} | {sp} | {s['n']} (seeds {','.join(map(str, s['seeds']))}) | {f4(s['mean'])} ({rng}) | "
                     f"{f4(s['recall'])} | {s['ece']:.4f} | {s['mb']:.1f} | {s['ips']:.0f} | {s['minutes']:.1f} | {'/'.join(s['roles'])} |")

base = stats("mobilenet_v2", "clean")
L += ["", f"### Ablations (MobileNetV2, clean split, seed 0)", "",
      f"Baseline = the {base['n']}-seed MobileNetV2 clean runs: mean {f4(base['mean'])}, range {f4(base['min'])}-{f4(base['max'])}. "
      "Each ablation is ONE seed and changes one factor.", "",
      "| variant | what changed | val acc | vs baseline mean | inside baseline seed range? |", "|---|---|---|---|---|"]
what = {"frozen_only": "backbone frozen for all 6 epochs (no fine-tuning)", "no_aug": "no online augmentation",
        "size128": "input 128 px", "size160": "input 160 px", "sched_constant": "constant fine-tuning LR",
        "sched_step": "step LR decay instead of cosine"}
abl = {}
for (arch, sp, tag), g in sorted(groups.items()):
    if arch == "mobilenet_v2" and sp == "clean" and tag != "base":
        a = g[0]["metrics"]["val_acc"]
        abl[tag] = a
        inside = base["min"] <= a <= base["max"]
        L.append(f"| {tag} | {what.get(tag, tag)} | {f4(a)} | {pp(a - base['mean'])} | {'yes' if inside else 'no (above)' if a > base['max'] else 'no (below)'} |")

L += ["", "### Near-duplicate leakage measured with embeddings (original split)", "",
      f"Share of the {leak['centered']['valid_to_train']['n']} validation images whose best match in train has cosine similarity >= threshold "
      f"(MobileNetV2 ImageNet features, mean-centered). Controls: train-to-train (hides self-match), best match among OTHER classes "
      f"(known non-copies), and the {leak['centered']['control_test_to_train']['n']} unlabeled test images. "
      f"Derived threshold (rule fixed before the run): A = {leak['centered']['derived_threshold']['A_other_class_p99.9']:.4f}, "
      f"B = {leak['centered']['derived_threshold']['B_histogram_valley']:.4f}, final = {leak['centered']['derived_threshold']['final']:.4f}.", "",
      "| threshold | val to train | train to train | other-class control | test (33) to train |", "|---|---|---|---|---|"]
c = leak["centered"]
for k in sorted(c["valid_to_train"]["frac_at_least"], key=float):
    row = [c[n]["frac_at_least"].get(k) for n in ("valid_to_train", "control_train_to_train", "control_valid_to_train_other_class", "control_test_to_train")]
    L.append(f"| {k} | " + " | ".join(pct(v) if v is not None else "-" for v in row) + " |")
q = c["valid_to_train"]["quantiles"]
L += ["", f"Best-match similarity quantiles (val to train, centered): " + ", ".join(f"p{float(k) * 100:g} = {v:.3f}" for k, v in q.items())
      + ". The distribution is a continuum, not two clean modes.",
      f"Raw (uncentered) features, for comparison: derived threshold {leak['raw']['derived_threshold']['final']:.4f}; "
      + ", ".join(f">= {k}: {pct(v)}" for k, v in leak["raw"]["valid_to_train"]["frac_at_least"].items() if k in ("0.9", "0.95", "0.98")) + ".", ""]

L += ["### Building the clean split", "", f"Rule: start from the derived threshold; if the grouping is degenerate, try stricter fixed ones. "
      f"Chosen threshold: **{split['threshold']}** (source: {split['threshold_source']}, healthy: {split['split_is_healthy']}).", "",
      "| threshold | accepted | reason if rejected | groups | largest group | val share per class (min-max) | original-split val with near copy |",
      "|---|---|---|---|---|---|---|"]
for a in split["attempts"] + split["sensitivity"]:
    sh = a.get("val_share_per_class_min_max")
    L.append(f"| {a['threshold']} | {'yes' if a['ok'] else 'no'}{' (sensitivity only)' if a in split['sensitivity'] else ''} | "
             f"{'; '.join(a['reasons']) or '-'} | {a.get('n_groups', '-')} | {a.get('max_group', '-')} | "
             f"{f'{sh[0]:.3f}-{sh[1]:.3f}' if sh else '-'} | {pct(a['original_split_val_with_near_copy_fraction']) if 'original_split_val_with_near_copy_fraction' in a else '-'} |")
cs, og = split["clean_split"], split["original_split"]
L += ["", f"Clean split: {cs['n_train']} train / {cs['n_val']} val ({pct(cs['val_fraction_overall'])} val; per-class "
      f"{cs['val_fraction_per_class_min_max'][0]:.3f}-{cs['val_fraction_per_class_min_max'][1]:.3f}). "
      f"Validation images with an embedding near-copy (>= {split['threshold']}) in train: original split {og['val_with_near_copy_in_train']} of {og['n_val']} "
      f"({pct(og['fraction'])}); clean split {cs['val_with_near_copy_in_train']}.", ""]

if xc:
    o, k, g = xc["original_split"], xc["clean_split_used_in_this_repo"], xc["embedding_groups"]
    L += ["### Cross-check with file names (extra)", "", f"Assumption: {xc['assumption']}.", "",
          "| split | val images | with a same-source image in train | share of all val | share of val with a parsable id |", "|---|---|---|---|---|",
          f"| original | {o['n_val']} | {o['with_sibling_in_train']} | {pct(o['fraction_of_all_val'])} | {pct(o['fraction_of_keyed_val'])} |",
          f"| clean (this repo) | {k['n_val']} | {k['with_sibling_in_train']} | {pct(k['fraction_of_all_val'])} | {pct(k['fraction_of_keyed_val'])} |", "",
          f"{xc['n_images_with_key']} of {xc['n_images']} files have a parsable source id; {g['n_sources']} distinct sources, "
          f"{g['images_per_source_mean']:.2f} images per source on average (max {g['images_per_source_max']}); "
          f"{g['sources_spanning_multiple_labels']} sources span more than one label. The embedding groups kept "
          f"{g['sources_kept_in_one_group']} of {g['sources_with_more_than_one_image']} multi-image sources whole, and "
          f"{g['images_grouped_with_a_sibling']} of {g['images_with_a_sibling']} images with a sibling share a group with one.", "",
          "Which augmentations the embedding grouping caught (images that have a same-source sibling; top types):", "",
          "| file-name suffix | images with a sibling | grouped with a sibling | share |", "|---|---|---|---|"]
    for a, v in list(g["captured_by_augmentation"].items())[:8]:
        L.append(f"| {a} | {v['n_images_with_sibling']} | {v['grouped_with_a_sibling']} | {pct(v['grouped_with_a_sibling'] / v['n_images_with_sibling'])} |")
    L.append("")

d = yolo["detector"]
L += ["### Detector stage and background-only check", "",
      f"Detector: {d['weights']} (sha256 {d['weights_sha256'][:12]}..., ultralytics {d['ultralytics']}, {d['license']}), {d['kind']}, "
      f"class filter {d['class_filter']} (COCO 'potted plant'), conf {d['conf']}. On {yolo['n_images']} clean-val images it found a box in "
      f"{yolo['n_detected']} ({pct(1 - yolo['no_detection_rate'])}); no-detection rate {pct(yolo['no_detection_rate'])}. "
      f"Classifier accuracy on that sample: whole image {f4(yolo['acc_whole_all'])}, two-stage with fallback {f4(yolo['acc_two_stage_with_fallback_all'])}."
      + (f" On the {yolo['n_detected']} detected images: whole {f4(yolo['on_detected_subset']['acc_whole'])}, crop {f4(yolo['on_detected_subset']['acc_crop'])}."
         if yolo["on_detected_subset"] else ""), "",
      f"Background-only check (EXTRA): an MLP on the colour statistics of four 16x16 image corners reaches "
      f"{pct(bg['original']['corner_only_val_acc'])} on the original split and {pct(bg['clean']['corner_only_val_acc'])} on the clean split "
      f"(uniform chance {pct(bg['original']['chance_uniform'])}, majority class {pct(bg['original']['majority_class_val_acc'])}).", ""]

o_m, c_m = stats("mobilenet_v2", "original"), base
e_o, e_c = stats("efficientnet_b3", "original"), stats("efficientnet_b3", "clean")
rel_ = lambda x, s: "inside" if s["min"] <= x <= s["max"] else "outside"  # noqa: E731
obs = [f"MobileNetV2: clean-split mean {f4(c_m['mean'])} (n={c_m['n']}) vs original-split single run {f4(o_m['mean'])}; difference {pp(c_m['mean'] - o_m['mean'])}; "
       f"the original-split value is {rel_(o_m['mean'], c_m)} the clean 3-seed range ({f4(c_m['min'])}-{f4(c_m['max'])}).",
       f"EfficientNet-B3: clean-split mean {f4(e_c['mean'])} (n={e_c['n']}) vs original-split single run {f4(e_o['mean'])}; difference {pp(e_c['mean'] - e_o['mean'])}; "
       f"the original-split value is {rel_(e_o['mean'], e_c)} the clean 3-seed range ({f4(e_c['min'])}-{f4(e_c['max'])}).",
       f"EfficientNet-B3 vs MobileNetV2 on the clean split: {pp(e_c['mean'] - c_m['mean'])}; seed ranges "
       f"{'overlap' if max(c_m['min'], e_c['min']) <= min(c_m['max'], e_c['max']) else 'do not overlap'}. B3 is {e_c['mb'] / c_m['mb']:.1f}x the size and "
       f"{c_m['ips'] / e_c['ips']:.1f}x slower in forward-only throughput (different input sizes: {e_c['img']} vs {c_m['img']} px)."]
for tag, a in abl.items():
    obs.append(f"Ablation {tag}: {f4(a)}, {pp(a - c_m['mean'])} vs baseline mean, {'inside' if c_m['min'] <= a <= c_m['max'] else 'outside'} the baseline seed range (single seed).")
if xc:
    obs.append(f"Embedding estimate (threshold {split['threshold']}): {pct(og['fraction'])} of original-split validation images have a near-copy in train; "
               f"the file-name cross-check finds a same-source image in train for {pct(xc['original_split']['fraction_of_all_val'])} (original) and "
               f"{pct(xc['clean_split_used_in_this_repo']['fraction_of_all_val'])} (this repo's clean split).")
obs.append(f"Detector: found a plant-like box in {yolo['n_detected']} of {yolo['n_images']} images.")
obs.append(f"A classifier on four image corners alone reaches {pct(bg['clean']['corner_only_val_acc'])} on the clean split (chance {pct(bg['clean']['chance_uniform'])}).")
L += ["### Observations (generated from the numbers above)", ""] + [f"- {x}" for x in obs] + [""]

text = "\n".join(L)
Path(args.out).write_text(text, encoding="utf-8")
print(text)
if args.readme:
    p = Path(args.readme)
    t = p.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.S)
    assert pat.search(t), "README has no RESULTS markers"
    p.write_text(pat.sub(lambda m: f"{m.group(1)}\n_Generated by `scripts/10_results_table.py` from `results/`; do not edit by hand._\n\n{text}\n{m.group(2)}", t), encoding="utf-8")
    print("updated", args.readme)
