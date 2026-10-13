# AgriScan: plant disease detection (vision + web re-implementation)

> **Re-implementation of a 2026 team project; original files lost; this repo covers my vision and web part.**

**Status: work in progress. No results yet.** Every number in this README will come from a run in this repo, saved as JSON under `results/` with seed, config and date. Nothing here is copied from the original project or from a CV.

## Credits

The original AgriScan was a team project (April 2026) in which I was lead developer, built together with:
**[TEAMMATE NAMES: to be filled in by Kabil]**.
The Arduino/Bluetooth robot hardware was built by the team and is **not** part of this repo.

## What this repo covers

- A two-stage vision pipeline: a pretrained YOLO detector finds plants in a frame, then a CNN classifies the disease of each crop (38 classes).
- Transfer-learning classifiers (MobileNetV2, EfficientNet-B3), trained and compared here.
- A small FastAPI web app: photo upload, single-frame endpoint, results page.
- A webcam or video file stands in for the robot's phone-camera stream.

The original project used both PyTorch and TensorFlow. **This re-implementation uses PyTorch only.**

## How this was built

I rebuilt this with Claude Code acting as a tutor: it explained each step, wrote code, and quizzed me before moving on. I worked through the code and notes so that I can explain it. The commit history shows the step-by-step process.

## Results

*TBD. Will list the original dataset split and a leakage-free ("clean") split side by side.*

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (use source .venv/bin/activate on Linux/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU only
pip install -r requirements.txt
python scripts/smoke_test.py  # 1 training step on random data, checks the install
```

Real training runs on a free Kaggle GPU: open `notebooks/kaggle_train.ipynb` in Kaggle (Internet on, GPU on); it clones this repo.
No credentials are stored in this repo. Data, model weights and `kaggle.json` are git-ignored.

## Limitations

*TBD (step 7).*

## Licenses and third-party models

- Ultralytics YOLO is licensed AGPL-3.0. *Exact detector, source, version and license: TBD (step 5).*
- Dataset: Kaggle "New Plant Diseases Dataset (Augmented)". *Source and license: to be verified (step 1).*
