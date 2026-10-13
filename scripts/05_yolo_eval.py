"""Step 5: does a pretrained detector help? Measured on the CLEAN validation split.

For each sampled validation image we compare, with the same classifier:
  whole  : classify the full image
  crop   : classify the crop of the highest-confidence detected box (only when something is detected)
  two-stage : crop if detected, else whole image (what the pipeline does)
and report how often the detector finds nothing. No claim of improvement is made unless the numbers show it.
"""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agriscan.data import find_dataset_root
from agriscan.infer import Predictor
from agriscan.pipeline import COCO_POTTED_PLANT, YoloDetector, crop
from agriscan.train import load_split
from agriscan.utils import save_results, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="classifier checkpoint (.pt) from 04_train --save-model")
ap.add_argument("--detector", default="yolo11n.pt", help="Ultralytics weights; downloaded at run time from the official release")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--classes", default=str(COCO_POTTED_PLANT), help="comma-separated detector class ids; 'all' = no filter")
ap.add_argument("--root", default="auto")
ap.add_argument("--csv", default="splits/clean_split.csv.gz")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--results", default="results/step5_yolo_eval.json")
args = ap.parse_args()

set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
root = find_dataset_root() if args.root == "auto" else Path(args.root)
_, va, _ = load_split("clean", root, args.csv)
va = va.sample(min(args.n, len(va)), random_state=args.seed).reset_index(drop=True)

clf = Predictor(args.model, device)
classes = None if args.classes == "all" else [int(c) for c in args.classes.split(",")]
det = YoloDetector(args.detector, conf=args.conf, classes=classes, device=device)
ck = getattr(det.model, "ckpt_path", None) or args.detector
sha = hashlib.sha256(Path(ck).read_bytes()).hexdigest() if Path(ck).is_file() else None

rows = []
for path, label in zip(va.path, va.label):
    im = Image.open(path).convert("RGB")
    boxes = sorted(det(im), key=lambda b: -b.conf)
    whole = clf.predict([im], 1)[0][0][0]
    cropped = clf.predict([crop(im, boxes[0])], 1)[0][0][0] if boxes else None
    rows.append({"found": bool(boxes), "whole_ok": whole == label, "crop_ok": None if cropped is None else cropped == label,
                 "frac_area": None if not boxes else (boxes[0].x2 - boxes[0].x1) * (boxes[0].y2 - boxes[0].y1) / (im.width * im.height)})

found = [r for r in rows if r["found"]]
n, k = len(rows), len(found)
two_stage_ok = [r["crop_ok"] if r["found"] else r["whole_ok"] for r in rows]
m = {"n_images": n, "no_detection_rate": 1 - k / n, "n_detected": k,
     "acc_whole_all": float(np.mean([r["whole_ok"] for r in rows])),
     "acc_two_stage_with_fallback_all": float(np.mean(two_stage_ok)),
     "on_detected_subset": None if not k else {
         "acc_whole": float(np.mean([r["whole_ok"] for r in found])),
         "acc_crop": float(np.mean([r["crop_ok"] for r in found])),
         "crop_right_whole_wrong": sum(r["crop_ok"] and not r["whole_ok"] for r in found),
         "whole_right_crop_wrong": sum(r["whole_ok"] and not r["crop_ok"] for r in found),
         "median_box_area_fraction": float(np.median([r["frac_area"] for r in found]))},
     "detector": {"weights": args.detector, "weights_sha256": sha, "ultralytics": version("ultralytics"),
                  "conf": args.conf, "class_filter": classes, "license": "AGPL-3.0 (Ultralytics)",
                  "kind": "generic COCO detector, not trained on leaves" if args.detector.startswith("yolo") else "custom"}}
print(json.dumps(m, indent=2))
save_results(args.results, name="step5_yolo_eval", seed=args.seed,
             config={"model": args.model, "csv": args.csv, "n": args.n, "root": str(root)}, metrics=m)
