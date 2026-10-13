"""Collect results/runs/*.json into results/summary.json and results/summary.md (README tables come from here)."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--out", default="results")
args = ap.parse_args()

runs = [json.loads(p.read_text()) for p in sorted(Path(args.runs).glob("*.json"))]
rows = []
for r in runs:
    c, m = r["config"], r["metrics"]
    rows.append({"name": r["name"], "role": c["role"], "arch": c["arch"], "split": c["split"], "seed": r["seed"],
                 "tag": c["tag"], "epochs": f"{c['epochs_head']}+{c['epochs_ft']}", "augment": c["augment"],
                 "img_size": m["img_size"], "schedule": c["schedule"], "val_acc": m["val_acc"],
                 "macro_recall": m["macro_recall"], "ece": m["ece_15bins"], "minutes": m["train_seconds"] / 60,
                 "img_per_s": m["forward_img_per_s"], "speed_device": m["speed_device"],
                 "size_mb": m["model_size_mb_fp32"], "date": r["date_utc"][:10], "commit": r["env"]["git_commit"]})

# Group by everything except the seed, and report mean and spread over however many seeds exist.
groups = defaultdict(list)
for x in rows:
    groups[(x["arch"], x["split"], x["tag"])].append(x)
agg = []
for (arch, split, tag), g in sorted(groups.items()):
    acc = np.array([x["val_acc"] for x in g])
    agg.append({"arch": arch, "split": split, "tag": tag, "n_seeds": len(g), "seeds": sorted(x["seed"] for x in g),
                "val_acc_mean": float(acc.mean()), "val_acc_min": float(acc.min()), "val_acc_max": float(acc.max()),
                "val_acc_std": float(acc.std(ddof=1)) if len(g) > 1 else None,
                "macro_recall_mean": float(np.mean([x["macro_recall"] for x in g])),
                "roles": sorted({x["role"] for x in g})})

Path(args.out).mkdir(parents=True, exist_ok=True)
(Path(args.out) / "summary.json").write_text(json.dumps({"runs": rows, "grouped": agg}, indent=2))

lines = ["| backbone | split | variant | seeds (n) | val acc (mean, min-max) | macro recall | role |", "|---|---|---|---|---|---|---|"]
for a in agg:
    spread = f"{a['val_acc_min']:.4f}-{a['val_acc_max']:.4f}" if a["n_seeds"] > 1 else "single run"
    lines.append(f"| {a['arch']} | {a['split']} | {a['tag']} | {a['n_seeds']} | {a['val_acc_mean']:.4f} ({spread}) | "
                 f"{a['macro_recall_mean']:.4f} | {', '.join(a['roles'])} |")
lines += ["", "| run | acc | ECE | train min | fwd img/s (device) | size MB | date | commit |", "|---|---|---|---|---|---|---|---|"]
for x in rows:
    lines.append(f"| {x['name']} | {x['val_acc']:.4f} | {x['ece']:.4f} | {x['minutes']:.1f} | "
                 f"{x['img_per_s']:.0f} ({x['speed_device']}) | {x['size_mb']:.1f} | {x['date']} | {x['commit']} |")
(Path(args.out) / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n{len(rows)} runs -> {args.out}/summary.json, summary.md")
