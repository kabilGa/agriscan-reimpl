"""Two-stage pipeline: a detector finds regions, each region is cropped and classified."""
from dataclasses import dataclass

from PIL import Image, ImageDraw

COCO_POTTED_PLANT = 58  # the only plant-like class of a generic COCO detector


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    name: str = "plant"


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model (AGPL-3.0). Weights are downloaded at run time."""

    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.25, classes=(COCO_POTTED_PLANT,), device=None):
        from ultralytics import YOLO, settings  # imported lazily: heavy, and only needed when a detector is used
        settings.update({"sync": False})  # Ultralytics sends anonymous usage analytics by default; switch that off
        self.model, self.conf, self.classes, self.device = YOLO(weights), conf, list(classes) if classes else None, device

    def __call__(self, image: Image.Image) -> list[Box]:
        # conf: drop boxes below this score. NMS (inside predict) removes overlapping duplicates.
        r = self.model.predict(image, conf=self.conf, classes=self.classes, device=self.device, verbose=False)[0]
        return [Box(*map(float, b.xyxy[0]), float(b.conf[0]), r.names[int(b.cls[0])]) for b in r.boxes]


def crop(image: Image.Image, b: Box, pad: float = 0.05) -> Image.Image:
    w, h = image.size
    dx, dy = (b.x2 - b.x1) * pad, (b.y2 - b.y1) * pad
    return image.crop((max(0, int(b.x1 - dx)), max(0, int(b.y1 - dy)), min(w, int(b.x2 + dx)), min(h, int(b.y2 + dy))))


def two_stage(image: Image.Image, detector, predictor, k: int = 3, fallback_whole: bool = True) -> dict:
    """Detect -> crop -> classify. If nothing is detected, optionally classify the whole image.

    Returns {"detections": [{"box", "det_conf", "top": [(label, prob)...]}], "used_fallback": bool}.
    """
    boxes = detector(image) if detector is not None else []
    used_fallback = False
    if not boxes and fallback_whole:
        w, h = image.size
        boxes, used_fallback = [Box(0, 0, w, h, 1.0, "whole image")], True
    tops = predictor.predict([crop(image, b) for b in boxes], k) if boxes else []
    return {"used_fallback": used_fallback,
            "detections": [{"box": [b.x1, b.y1, b.x2, b.y2], "det_conf": b.conf, "det_name": b.name, "top": t}
                           for b, t in zip(boxes, tops)]}


def draw(image: Image.Image, result: dict) -> Image.Image:
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for det in result["detections"]:
        x1, y1, x2, y2 = det["box"]
        label, p = det["top"][0]
        d.rectangle((x1, y1, x2, y2), outline=(0, 200, 0), width=3)
        d.text((x1 + 4, y1 + 4), f"{label} {p:.0%}", fill=(255, 255, 0))
    return out
