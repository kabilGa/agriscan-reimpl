"""Minimal FastAPI demo: upload a leaf photo (or a video frame), get the top-3 classes.

Configuration comes from environment variables only:
  AGRISCAN_MODEL     path to a checkpoint written by train.save_checkpoint (without it, /predict answers 503)
  AGRISCAN_DETECTOR  optional Ultralytics weights name/path (e.g. yolo11n.pt) enabling the crop-first mode
"""
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

NOTE = ("Demo trained on lab-style leaf photos (PlantVillage-like). Not a diagnosis: do not base treatment "
        "decisions on it. Probabilities are raw softmax scores and are not calibrated.")
MAX_BYTES = 10_000_000
MAX_PIXELS = 25_000_000

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AgriScan demo</title>
<style>body{font:16px system-ui;max-width:42rem;margin:2rem auto;padding:0 1rem}
.warn{background:#fff3cd;border:1px solid #e0c36a;padding:.7rem;border-radius:6px}
pre{background:#f4f4f4;padding:.7rem;overflow:auto}</style>
<h1>AgriScan demo</h1>
<p class=warn id=note></p>
<p><input type=file id=f accept="image/*"></p>
<p><label><input type=checkbox id=det> crop with the detector first (needs AGRISCAN_DETECTOR on the server)</label></p>
<p><button id=go>Classify</button></p>
<pre id=out>No result yet.</pre>
<script>
document.getElementById('note').textContent = %NOTE%;
document.getElementById('go').onclick = async () => {
  const file = document.getElementById('f').files[0], out = document.getElementById('out');
  if (!file) { out.textContent = 'Choose an image first.'; return; }
  const fd = new FormData(); fd.append('file', file);
  const url = '/predict?use_detector=' + document.getElementById('det').checked;
  const r = await fetch(url, {method: 'POST', body: fd});
  out.textContent = JSON.stringify(await r.json(), null, 2);   // textContent, never innerHTML
};
</script>"""


def _read_image(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_BYTES // 1_000_000} MB.")
    try:
        im = Image.open(io.BytesIO(data))
        if im.size[0] * im.size[1] > MAX_PIXELS:
            raise HTTPException(413, "Image has too many pixels.")
        im = im.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the upload as an image (use JPEG or PNG).")
    im.thumbnail((1600, 1600))  # bound the compute spent on one request
    return im


def _fmt(result: dict) -> dict:
    """Same {label, probability} shape for every endpoint."""
    dets = [{**d, "top": [{"label": l, "probability": p} for l, p in d["top"]]} for d in result["detections"]]
    return {"note": NOTE, "used_fallback_whole_image": result["used_fallback"], "detections": dets}


def create_app(model_path: str | None = None, detector_weights: str | None = None) -> FastAPI:
    model_path = model_path or os.environ.get("AGRISCAN_MODEL")
    detector_weights = detector_weights or os.environ.get("AGRISCAN_DETECTOR")
    state = {"predictor": None, "detector": None, "model_error": None, "detector_error": None}

    if model_path:
        try:
            from agriscan.infer import Predictor
            state["predictor"] = Predictor(model_path)
        except Exception as e:  # noqa: BLE001  keep the server up; /health and 503 explain what is wrong
            state["model_error"] = f"{type(e).__name__}: {e}"

    def get_detector():
        if state["detector"] is None and detector_weights and not state["detector_error"]:
            try:
                from agriscan.pipeline import YoloDetector
                state["detector"] = YoloDetector(detector_weights)
            except Exception as e:  # noqa: BLE001
                state["detector_error"] = f"{type(e).__name__}: {e}"
        return state["detector"]

    def need_model():
        if state["predictor"] is None:
            why = state["model_error"] or "AGRISCAN_MODEL is not set"
            raise HTTPException(503, f"Model not loaded ({why}). Set AGRISCAN_MODEL to a trained checkpoint (.pt) "
                                     "and restart the server. No weights are shipped with this repo.")
        return state["predictor"]

    app = FastAPI(title="AgriScan demo")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["predictor"] is not None, "model_error": state["model_error"],
                "detector_configured": bool(detector_weights), "detector_error": state["detector_error"], "note": NOTE}

    @app.get("/", response_class=HTMLResponse)
    def index():
        import json
        return PAGE.replace("%NOTE%", json.dumps(NOTE))

    @app.post("/predict")
    def predict(file: UploadFile = File(...), use_detector: bool = False):
        """Photo upload. With use_detector=true the plant is detected and cropped before classification."""
        predictor = need_model()
        im = _read_image(file.file.read())
        if use_detector:
            from agriscan.pipeline import two_stage
            det = get_detector()
            if det is None:
                raise HTTPException(400, "Detector not available: " + (state["detector_error"] or "AGRISCAN_DETECTOR is not set"))
            return _fmt(two_stage(im, det, predictor))
        top = predictor.predict([im])[0]
        return {"note": NOTE, "top3": [{"label": l, "probability": p} for l, p in top]}

    @app.post("/predict_frame")
    def predict_frame(file: UploadFile = File(...)):
        """One video frame (JPEG). Uses the detector when configured, otherwise classifies the whole frame."""
        from agriscan.pipeline import two_stage
        predictor = need_model()
        return _fmt(two_stage(_read_image(file.file.read()), get_detector(), predictor))

    return app


app = create_app()
