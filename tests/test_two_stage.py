"""Detect -> crop -> classify logic with a fake detector and a fake classifier (no weights, no ultralytics)."""
from PIL import Image

from agriscan.pipeline import Box, crop, draw, two_stage


class FakeClassifier:
    def __init__(self):
        self.sizes = []

    def predict(self, images, k=3):
        self.sizes += [im.size for im in images]
        return [[("leaf_a", 0.9), ("leaf_b", 0.1)][:k] for _ in images]


def test_crop_is_clamped_to_the_image():
    im = Image.new("RGB", (100, 80))
    c = crop(im, Box(-20, -20, 500, 500, 0.9), pad=0.1)
    assert c.size == (100, 80)
    assert crop(im, Box(10, 10, 30, 50, 0.9), pad=0.0).size == (20, 40)


def test_each_detection_is_cropped_and_classified():
    clf, im = FakeClassifier(), Image.new("RGB", (200, 100))
    det = lambda _: [Box(0, 0, 50, 50, 0.9), Box(100, 10, 180, 90, 0.8)]  # noqa: E731
    r = two_stage(im, det, clf)
    assert not r["used_fallback"] and len(r["detections"]) == 2
    assert clf.sizes[0] == (52, 52)   # 50 px box + 5% padding (2 px), clamped at the left/top border
    assert r["detections"][0]["top"][0][0] == "leaf_a"


def test_no_detection_falls_back_to_the_whole_image_and_says_so():
    clf, im = FakeClassifier(), Image.new("RGB", (64, 48))
    r = two_stage(im, lambda _: [], clf)
    assert r["used_fallback"] and clf.sizes == [(64, 48)] and r["detections"][0]["det_name"] == "whole image"
    assert two_stage(im, lambda _: [], clf, fallback_whole=False)["detections"] == []


def test_draw_returns_an_image_of_the_same_size():
    im = Image.new("RGB", (64, 48))
    out = draw(im, two_stage(im, lambda _: [Box(5, 5, 40, 40, 0.9)], FakeClassifier()))
    assert out.size == im.size and out.getpixel((5, 20)) != (0, 0, 0)   # a box outline was painted
