"""Run the two-stage pipeline on an image, a video file or a webcam, and write the annotated result.

  python scripts/06_pipeline_demo.py --model m.pt --source leaf.jpg --out out.png
  python scripts/06_pipeline_demo.py --model m.pt --source clip.mp4 --out out.mp4 --detector yolo11n.pt
  python scripts/06_pipeline_demo.py --model m.pt --source 0 --out cam.mp4 --max-frames 200   # webcam index 0
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from agriscan.infer import Predictor
from agriscan.pipeline import YoloDetector, draw, two_stage

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--source", required=True, help="image path, video path, or webcam index such as 0")
ap.add_argument("--out", required=True)
ap.add_argument("--detector", default="none", help="Ultralytics weights (e.g. yolo11n.pt) or 'none' to classify whole frames")
ap.add_argument("--every", type=int, default=5, help="classify every N-th video frame (others reuse the last result)")
ap.add_argument("--max-frames", type=int, default=0)
args = ap.parse_args()

clf = Predictor(args.model)
det = None if args.detector == "none" else YoloDetector(args.detector)

if Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png"}:
    im = Image.open(args.source).convert("RGB")
    res = two_stage(im, det, clf)
    draw(im, res).save(args.out)
    print("fallback to whole image:", res["used_fallback"])
    for d in res["detections"]:
        print(d["det_name"], [round(c) for c in d["box"]], [(l, round(p, 3)) for l, p in d["top"]])
else:
    import cv2  # video I/O only; all model code works on PIL images
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source {args.source}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 15, (w, h))
    i, res = 0, None
    while cap.isOpened() and (not args.max_frames or i < args.max_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # OpenCV is BGR, PIL/torch expect RGB
        if i % args.every == 0 or res is None:
            res = two_stage(im, det, clf)
        writer.write(cv2.cvtColor(np.array(draw(im, res)), cv2.COLOR_RGB2BGR))
        i += 1
    cap.release()
    writer.release()
    print(f"wrote {i} frames to {args.out}")
