Re-implementation of a 2026 team project; original files lost; this repo covers my vision and web part.

# AgriScan: plant disease detection (vision + web re-implementation)

**Status: code written and tested on synthetic data only. Nothing has been run on the real dataset or a GPU yet, so this README contains no results.** Every number that will appear here comes from a run in this repo, saved as JSON under `results/` with seed, config, date and code commit. Nothing is copied from the original project or from a CV.

## Status

A step moves to "ran on Kaggle" only when its results JSON exists in `results/`.

| Step | What | Status |
|---|---|---|
| 0 | Environment, repo, CPU smoke test, results/seed helpers | code written and tested on synthetic data (CPU smoke test run locally) |
| 1 | Data inspection, near-duplicate leakage measurement, clean grouped split | code written and tested on synthetic data |
| 2 | MobileNetV2: frozen head, then fine-tuning (original and clean split) | code written and tested on synthetic data |
| 3 | EfficientNet-B3 (original and clean split), comparison | code written and tested on synthetic data |
| 4 | Ablations: frozen vs fine-tuned, augmentation, image size, LR schedule | code written and tested on synthetic data |
| 5 | YOLO stage (generic pretrained detector) and two-stage pipeline | code written and tested with a fake detector; see "Detector stage" |
| 6 | FastAPI app (photo upload, frame endpoint, page) | code written; tested with a dummy model, no real weights |
| 7 | Evaluation and limits from real results | not started (needs the Kaggle results) |
| 8 | Final docs | not started (README results table needs real results) |
| optional | background-only check, ResNet-50 baseline | code written and tested on synthetic data; YOLO fine-tuning: not planned (no box labels) |

## Credits

The original AgriScan was a team project (April 2026) in which I was lead developer, built together with my teammates (their names will be added once they have agreed). The Arduino/Bluetooth robot hardware was built by the team and is **not** part of this repo.

## What this repo covers

- A two-stage vision pipeline: a pretrained YOLO detector finds regions, each is cropped and classified by a CNN (38 classes).
- Transfer-learning classifiers (MobileNetV2, EfficientNet-B3), trained and compared, on the dataset's original split **and** on a leakage-free split I built.
- A small FastAPI web app: photo upload, single-frame endpoint, results page.
- A webcam or video file stands in for the robot's phone-camera stream (`scripts/06_pipeline_demo.py`).

The original project used both PyTorch and TensorFlow. **This re-implementation uses PyTorch only.**

## Detector stage (read this)

I did not train any detector. The stage uses the generic, official **Ultralytics YOLO11n** (COCO classes; weights downloaded at run time from the official release, never committed), restricted to COCO's "potted plant" class. It is **not a leaf-specific model**: the stage demonstrates the pipeline (detect, crop, classify) and is not a trained component. Whether it helps is measured on the clean validation split by `scripts/05_yolo_eval.py` (including how often it finds nothing); no improvement is claimed unless that measurement shows it.

Two Hugging Face "leaf detection" models were considered and **not used** because their licence could not be confirmed (see DECISIONS.md, 5.1).

## How this was built

Written with substantial help from Claude Code, acting as tutor and coding assistant (see the `Co-Authored-By` trailers in the commit history). Design choices and their alternatives are in `DECISIONS.md`.

## Setup and tests

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on Linux/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU only
pip install -e ".[dev,app]"       # add ",detect" for the YOLO stage (Ultralytics, AGPL-3.0)
python scripts/smoke_test.py   # 1 training step on random data, checks the install
python -m pytest -q tests      # logic tests on synthetic data (no downloads, no GPU needed)
```

Real experiments run on a free Kaggle GPU: see `KAGGLE_RUN.md` (one notebook, "Run all"). No credentials are stored in this repo; data, model weights and `kaggle.json` are git-ignored.

## Run the web app

```bash
set AGRISCAN_MODEL=path\to\checkpoint.pt      # Windows; export AGRISCAN_MODEL=... elsewhere
set AGRISCAN_DETECTOR=yolo11n.pt              # optional: crop-first mode (needs ultralytics)
uvicorn agriscan.app:app --port 8000
```

Without `AGRISCAN_MODEL` the server still starts: `/health` reports it and `/predict` answers 503 with an explanation. The page states that this is a demo trained on lab-style images, not a diagnosis.

## Results

*None yet. Will list the original dataset split and the leakage-free ("clean") split side by side, plus ablations, detector measurements, and per-class results, all from `results/`.*

## Limitations (preliminary, to be completed with real results)

- Training images are lab-style leaf photos; field photos will behave differently (domain shift).
- The original dataset split may leak near-duplicates; the leakage estimate is a lower bound and depends on a similarity threshold.
- No separate test set: hyperparameters were fixed in advance, but ablations are compared on validation.
- Softmax probabilities are uncalibrated.
- The detector stage is generic, not leaf-specific.

## Licenses and third-party models

- Ultralytics YOLO is licensed AGPL-3.0; the YOLO11n weights are from the official Ultralytics release (exact file hash and package version are recorded in `results/step5_yolo_eval.json` after a run).
- Dataset: Kaggle "New Plant Diseases Dataset (Augmented)". *Licence: to be read from the dataset page and recorded here before any publication.*
- torchvision ImageNet weights (MobileNetV2, EfficientNet-B3, ResNet-50) are downloaded at run time.
