"""Kaggle bundle contents, and the FastAPI app (503 without weights, predictions with a tiny dummy model)."""
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_has_code_only(tmp_path):
    out = tmp_path / "b.zip"
    r = subprocess.run([sys.executable, "scripts/make_kaggle_bundle.py", "--out", out], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    names = zipfile.ZipFile(out).namelist()
    assert "src/agriscan/train.py" in names and "BUNDLE_INFO.json" in names and "pyproject.toml" in names
    for n in names:
        parts = Path(n).parts
        assert Path(n).name not in {"LEARNING_NOTES.md", "INTERVIEW_QA.md", "kaggle.json"}, n
        assert not ({".git", ".venv", "artifacts", "results", "splits"} & set(parts)), n
        assert Path(n).suffix not in {".pt", ".npz", ".gz", ".jpg", ".png", ".zip"}, n
    assert "commit" in json.loads(zipfile.ZipFile(out).read("BUNDLE_INFO.json"))


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agriscan.app import create_app  # noqa: E402
from agriscan.train import build_model, save_checkpoint  # noqa: E402


def _jpeg(size=80, color=(30, 160, 60)):
    b = io.BytesIO()
    Image.new("RGB", (size, size), color).save(b, "JPEG")
    return b.getvalue()


@pytest.fixture()
def dummy_model(tmp_path):
    torch.manual_seed(0)
    model, _, _ = build_model("mobilenet_v2", 4, pretrained=False)
    path = tmp_path / "dummy.pt"
    save_checkpoint(path, model, "mobilenet_v2", ["a", "b", "c", "d"], 64)
    return str(path)


def test_starts_without_weights_and_answers_503(monkeypatch):
    monkeypatch.delenv("AGRISCAN_MODEL", raising=False)
    c = TestClient(create_app())
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is False
    r = c.post("/predict", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 503 and "AGRISCAN_MODEL" in r.json()["detail"]
    assert c.post("/predict_frame", files={"file": ("x.jpg", _jpeg(), "image/jpeg")}).status_code == 503
    page = c.get("/")
    assert page.status_code == 200 and "Not a diagnosis" in page.text


def test_bad_model_path_does_not_crash_the_server():
    c = TestClient(create_app(model_path="/does/not/exist.pt"))
    h = c.get("/health").json()
    assert h["model_loaded"] is False and h["model_error"]
    assert c.post("/predict", files={"file": ("x.jpg", _jpeg(), "image/jpeg")}).status_code == 503


def test_predict_top3(dummy_model):
    c = TestClient(create_app(model_path=dummy_model))
    assert c.get("/health").json()["model_loaded"] is True
    r = c.post("/predict", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    top = r.json()["top3"]
    assert len(top) == 3 and "diagnosis" in r.json()["note"]
    ps = [t["probability"] for t in top]
    assert ps == sorted(ps, reverse=True) and 0 < sum(ps) <= 1.0 + 1e-6


def test_rejects_non_images_and_oversize(dummy_model):
    c = TestClient(create_app(model_path=dummy_model))
    assert c.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 400
    assert c.post("/predict", files={"file": ("big.jpg", b"0" * 10_000_001, "image/jpeg")}).status_code == 413


def test_detector_crop_path_and_frame_endpoint(dummy_model, monkeypatch):
    from agriscan import pipeline

    class FakeDetector:  # stands in for YOLO: one fixed box
        def __init__(self, *a, **k):
            pass

        def __call__(self, im):
            return [pipeline.Box(10, 10, 60, 60, 0.9, "potted plant")]

    monkeypatch.setattr(pipeline, "YoloDetector", FakeDetector)
    c = TestClient(create_app(model_path=dummy_model, detector_weights="fake.pt"))
    r = c.post("/predict?use_detector=true", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")}).json()
    assert r["used_fallback_whole_image"] is False and r["detections"][0]["det_name"] == "potted plant"
    f = c.post("/predict_frame", files={"file": ("frame.jpg", _jpeg(), "image/jpeg")}).json()
    top = f["detections"][0]["top"]
    assert len(top) == 3 and set(top[0]) == {"label", "probability"}


def test_detector_requested_but_not_configured(dummy_model):
    c = TestClient(create_app(model_path=dummy_model))
    assert c.post("/predict?use_detector=true", files={"file": ("l.jpg", _jpeg(), "image/jpeg")}).status_code == 400
