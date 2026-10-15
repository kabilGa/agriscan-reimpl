Re-implementation of a 2026 team project; original files lost; this repo covers my vision and web part.

# AgriScan: plant disease detection (vision + web re-implementation)

**Status: one full Kaggle run (2026-10-15, code commit `ace8631`) is recorded in `results/`. Read "Evaluation and limits" before quoting any number: the validation sets used here share source photos with the training data, so the high accuracies below measure in-dataset recall, not performance on new leaves or field photos.** Every number in this README is generated from JSON files under `results/` (each with seed, config, date and code commit) by `scripts/10_results_table.py`. Nothing is copied from the original project or from a CV, and no claim is made about the original project's reported accuracy.

## Status

A step is marked "ran on Kaggle" only because its results JSON exists in `results/`.

| Step | What | Status |
|---|---|---|
| 0 | Environment, repo, CPU smoke test, results/seed helpers | done; smoke test run locally on CPU |
| 1 | Data inspection, near-duplicate leakage measurement, clean grouped split | ran on Kaggle (2026-10-15, `ace8631`): `results/step1_*.json`, `splits/clean_split.csv.gz`. The measurement found that the "clean" split is not clean (see below) |
| 2 | MobileNetV2: frozen head, then fine-tuning (original and clean split) | ran on Kaggle (2026-10-15, `ace8631`): `results/runs/mobilenet_v2__*.json` |
| 3 | EfficientNet-B3 (original and clean split), comparison | ran on Kaggle (2026-10-15, `ace8631`): `results/runs/efficientnet_b3__*.json` |
| 4 | Ablations: frozen vs fine-tuned, augmentation, image size, LR schedule | ran on Kaggle (2026-10-15, `ace8631`), one seed each: `results/runs/mobilenet_v2__clean__s0__*.json` |
| 5 | YOLO stage (generic pretrained detector) and two-stage pipeline | ran on Kaggle (2026-10-15, `ace8631`): `results/step5_yolo_eval.json`. The detector found almost nothing, so the stage adds nothing measurable here |
| 6 | FastAPI app (photo upload, frame endpoint, page) | not a Kaggle step. Tested with TestClient and a dummy model, and started locally with the real MobileNetV2 checkpoint (12 sample training crops through `/predict`, a sanity check only). Weights are not in the repo |
| 7 | Evaluation and limits from real results | done: "Evaluation and limits" below |
| 8 | Final docs | done: README, DECISIONS.md, LICENSES.md |
| optional | background-only check, ResNet-50 baseline | ran on Kaggle (2026-10-15, `ace8631`): `results/extra_background_check.json`, `results/runs/resnet50__clean__s0__base.json`. YOLO fine-tuning: not done (no box labels) |
| extra | file-name cross-check of leakage and of the clean split | ran locally on the committed split file (commit recorded in the JSON): `results/extra_filename_crosscheck.json` |

## Credits

The original AgriScan was a team project (April 2026) in which I was lead developer, built together with my teammates Yassine Laabidi, Wassim Ferchichi and Mohamed Fedi Thabet (named here with their agreement). The Arduino/Bluetooth robot hardware was built by the team and is **not** part of this repo.

## What this repo covers

- A two-stage vision pipeline: a pretrained YOLO detector finds regions, each is cropped and classified by a CNN (38 classes).
- Transfer-learning classifiers (MobileNetV2, EfficientNet-B3), trained and compared on the dataset's original split and on a second split I built by grouping embedding near-copies.
- A small FastAPI web app: photo upload, single-frame endpoint, results page.
- A webcam or video file stands in for the robot's phone-camera stream (`scripts/06_pipeline_demo.py`).

The original project used both PyTorch and TensorFlow. **This re-implementation uses PyTorch only.**

## Detector stage (read this)

I did not train any detector. The stage uses the generic, official **Ultralytics YOLO11n** (COCO classes; weights downloaded at run time from the official release, never committed), restricted to COCO's "potted plant" class. It is **not a leaf-specific model**: it demonstrates the pipeline (detect, crop, classify) and is not a trained component. Measured on the clean validation split, it finds a plant-like box in almost none of these close-up leaf photos (exact rate in the generated table below), so in practice the pipeline falls back to classifying the whole image. No improvement from the detector is claimed.

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

Real experiments run on a free Kaggle GPU: see `KAGGLE_RUN.md` (one notebook, "Run all"; it also records what the first real run used). No credentials are stored in this repo; data, model weights, galleries and `kaggle.json` are git-ignored.

Rebuild the tables below from the committed JSON files:

```bash
python scripts/09_filename_crosscheck.py                 # extra: file-name based leakage check (needs only splits/clean_split.csv.gz)
python scripts/10_results_table.py --readme README.md    # rewrites the generated block between the RESULTS markers
```

## Run the web app

```bash
set AGRISCAN_MODEL=path\to\checkpoint.pt      # Windows; export AGRISCAN_MODEL=... elsewhere
set AGRISCAN_DETECTOR=yolo11n.pt              # optional: crop-first mode (needs ultralytics)
uvicorn agriscan.app:app --port 8000
```

Without `AGRISCAN_MODEL` the server still starts: `/health` reports it and `/predict` answers 503 with an explanation. The page states that this is a demo trained on lab-style images, not a diagnosis.

## Results

<!-- RESULTS:START -->
_Generated by `scripts/10_results_table.py` from `results/`; do not edit by hand._

### Run provenance

All 15 training runs and the step 1 / step 5 / extra stages come from one Kaggle session on Tesla T4 (torch 2.11.0+cu128, torchvision 0.26.0+cu128, ultralytics 8.4.177, Python 3.13.15), wall time 5.8 h, code commit ace8631, stages: 20 of 20 done. Dataset: 70295 train + 17572 valid images, 38 classes (per-class train 1642-2022), 33 unlabeled test images used only as a control.

### Classifiers: original split vs clean split

Final-epoch validation metrics (no best-epoch selection). 'Original' is the dataset's own train/valid split; 'clean' is the embedding-grouped split built here (see the leakage sections: it is NOT free of leakage). Speed = forward-only images/s on one batch on the GPU above; time = minutes per run.

| backbone | input px | split | seeds (n) | val acc, mean (min-max) | macro recall | ECE | size MB | fwd img/s | min/run | role |
|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 224 | clean | 3 (seeds 0,1,2) | 0.9909 (0.9898-0.9925) | 0.9908 | 0.0020 | 9.1 | 2094 | 18.2 | core/seed_replicate |
| mobilenet_v2 | 224 | original | 1 (seeds 0) | 0.9917 (single run) | 0.9915 | 0.0015 | 9.1 | 2097 | 18.7 | core |
| efficientnet_b3 | 300 | clean | 3 (seeds 0,1,2) | 0.9976 (0.9974-0.9978) | 0.9975 | 0.0009 | 43.0 | 371 | 31.8 | core/seed_replicate |
| efficientnet_b3 | 300 | original | 1 (seeds 0) | 0.9970 (single run) | 0.9968 | 0.0005 | 43.0 | 369 | 31.7 | core |
| resnet50 | 224 | clean | 1 (seeds 0) | 0.9961 (single run) | 0.9961 | 0.0011 | 94.3 | 801 | 25.4 | extra |

### Ablations (MobileNetV2, clean split, seed 0)

Baseline = the 3-seed MobileNetV2 clean runs: mean 0.9909, range 0.9898-0.9925. Each ablation is ONE seed and changes one factor.

| variant | what changed | val acc | vs baseline mean | inside baseline seed range? |
|---|---|---|---|---|
| frozen_only | backbone frozen for all 6 epochs (no fine-tuning) | 0.9373 | -5.36 pp | no (below) |
| no_aug | no online augmentation | 0.9965 | +0.56 pp | no (above) |
| sched_constant | constant fine-tuning LR | 0.9813 | -0.96 pp | no (below) |
| sched_step | step LR decay instead of cosine | 0.9924 | +0.15 pp | yes |
| size128 | input 128 px | 0.9658 | -2.51 pp | no (below) |
| size160 | input 160 px | 0.9856 | -0.54 pp | no (below) |

### Near-duplicate leakage measured with embeddings (original split)

Share of the 17572 validation images whose best match in train has cosine similarity >= threshold (MobileNetV2 ImageNet features, mean-centered). Controls: train-to-train (hides self-match), best match among OTHER classes (known non-copies), and the 33 unlabeled test images. Derived threshold (rule fixed before the run): A = 0.8021, B = 0.8046, final = 0.8046.

| threshold | val to train | train to train | other-class control | test (33) to train |
|---|---|---|---|---|
| 0.757 | 59.4% | 59.5% | 1.0% | 30.3% |
| 0.8 | 46.3% | 46.3% | 0.1% | 18.2% |
| 0.802 | 45.7% | 45.7% | 0.1% | 18.2% |
| 0.805 | 44.9% | 44.8% | 0.1% | 18.2% |
| 0.829 | 38.5% | 38.5% | 0.0% | 18.2% |
| 0.85 | 33.0% | 32.9% | 0.0% | 15.2% |
| 0.9 | 15.7% | 15.8% | 0.0% | 3.0% |
| 0.95 | 1.9% | 1.9% | 0.0% | 0.0% |
| 0.98 | 0.1% | 0.1% | 0.0% | 0.0% |
| 0.99 | 0.0% | 0.1% | 0.0% | 0.0% |

Best-match similarity quantiles (val to train, centered): p5 = 0.604, p25 = 0.706, p50 = 0.786, p75 = 0.876, p95 = 0.931. The distribution is a continuum, not two clean modes.
Raw (uncentered) features, for comparison: derived threshold 0.9407; >= 0.9: 87.9%, >= 0.95: 38.0%, >= 0.98: 1.9%.

### Building the clean split

Rule: start from the derived threshold; if the grouping is degenerate, try stricter fixed ones. Chosen threshold: **0.9** (source: fallback_ladder, healthy: True).

| threshold | accepted | reason if rejected | groups | largest group | val share per class (min-max) | original-split val with near copy |
|---|---|---|---|---|---|---|
| 0.8045563697814941 | no | giant group (1890 images) | 57238 | 1890 | 0.187-0.245 | 45.0% |
| 0.9 | yes | - | 79025 | 152 | 0.200-0.201 | 15.7% |
| 0.85 | no (sensitivity only) | giant group (1148 images) | 67936 | 1148 | 0.200-0.211 | 33.0% |
| 0.93 | yes (sensitivity only) | - | 84946 | 6 | 0.200-0.201 | 5.4% |

Clean split: 70279 train / 17588 val (20.0% val; per-class 0.200-0.201). Validation images with an embedding near-copy (>= 0.9) in train: original split 2762 of 17572 (15.7%); clean split 0.

### Cross-check with file names (extra)

Assumption: files sharing the UUID before '___' in their name are copies of one source photo (not verified against the dataset's documentation).

| split | val images | with a same-source image in train | share of all val | share of val with a parsable id |
|---|---|---|---|---|
| original | 17572 | 11083 | 63.1% | 64.8% |
| clean (this repo) | 17588 | 9536 | 54.2% | 55.7% |

85483 of 87867 files have a parsable source id; 44648 distinct sources, 1.91 images per source on average (max 15); 0 sources span more than one label. The embedding groups kept 4021 of 18634 multi-image sources whole, and 13984 of 59469 images with a sibling share a group with one.

Which augmentations the embedding grouping caught (images that have a same-source sibling; top types):

| file-name suffix | images with a sibling | grouped with a sibling | share |
|---|---|---|---|
| (none) | 17213 | 2869 | 16.7% |
| _180deg | 8535 | 2734 | 32.0% |
| _flipTB | 7142 | 2733 | 38.3% |
| _flipLR | 6268 | 2844 | 45.4% |
| _new30degFlipLR | 4946 | 21 | 0.4% |
| _90deg | 4107 | 46 | 1.1% |
| _270deg | 4107 | 12 | 0.3% |
| _newPixel25 | 2370 | 0 | 0.0% |

### Detector stage and background-only check

Detector: yolo11n.pt (sha256 0ebbc80d4a76..., ultralytics 8.4.177, AGPL-3.0 (Ultralytics)), generic COCO detector, not trained on leaves, class filter [58] (COCO 'potted plant'), conf 0.25. On 2000 clean-val images it found a box in 3 (0.1%); no-detection rate 99.9%. Classifier accuracy on that sample: whole image 0.9925, two-stage with fallback 0.9925. On the 3 detected images: whole 1.0000, crop 1.0000.

Background-only check (EXTRA): an MLP on the colour statistics of four 16x16 image corners reaches 63.3% on the original split and 63.9% on the clean split (uniform chance 2.6%, majority class 2.9%).

### Observations (generated from the numbers above)

- MobileNetV2: clean-split mean 0.9909 (n=3) vs original-split single run 0.9917; difference -0.08 pp; the original-split value is inside the clean 3-seed range (0.9898-0.9925).
- EfficientNet-B3: clean-split mean 0.9976 (n=3) vs original-split single run 0.9970; difference +0.06 pp; the original-split value is outside the clean 3-seed range (0.9974-0.9978).
- EfficientNet-B3 vs MobileNetV2 on the clean split: +0.67 pp; seed ranges do not overlap. B3 is 4.7x the size and 5.6x slower in forward-only throughput (different input sizes: 300 vs 224 px).
- Ablation frozen_only: 0.9373, -5.36 pp vs baseline mean, outside the baseline seed range (single seed).
- Ablation no_aug: 0.9965, +0.56 pp vs baseline mean, outside the baseline seed range (single seed).
- Ablation sched_constant: 0.9813, -0.96 pp vs baseline mean, outside the baseline seed range (single seed).
- Ablation sched_step: 0.9924, +0.15 pp vs baseline mean, inside the baseline seed range (single seed).
- Ablation size128: 0.9658, -2.51 pp vs baseline mean, outside the baseline seed range (single seed).
- Ablation size160: 0.9856, -0.54 pp vs baseline mean, outside the baseline seed range (single seed).
- Embedding estimate (threshold 0.9): 15.7% of original-split validation images have a near-copy in train; the file-name cross-check finds a same-source image in train for 63.1% (original) and 54.2% (this repo's clean split).
- Detector: found a plant-like box in 3 of 2000 images.
- A classifier on four image corners alone reaches 63.9% on the clean split (chance 2.6%).

<!-- RESULTS:END -->

## Evaluation and limits

### What was measured

- **Data and splits.** The Kaggle derivative of PlantVillage (38 classes, 256x256 lab-style leaf photos; counts in the provenance paragraph above). "Original" is the dataset's own train/valid split. "Clean" is a split I built: near-copies found with ImageNet MobileNetV2 embeddings (cosine similarity, mean-centered) are linked, and each connected group goes wholly to train or to validation.
- **Models.** MobileNetV2 (224 px), EfficientNet-B3 (300 px, the size in the weights' metadata) and, as an extra, ResNet-50. Each run: 2 epochs with the backbone frozen, then 4 epochs fine-tuning the last blocks; metrics are from the final epoch. No hyperparameter was tuned on validation, no checkpoint was selected on validation, and there is no separate test set (the 33 unlabeled test images were used only as a leakage control).
- **Seeds.** Seeds 0, 1 and 2 for both backbones on the clean split. Seed 0 only for the original split, for every ablation and for ResNet-50. So "original vs clean" compares one run against the range of three, and each ablation is a single run.
- **Thresholds.** The embedding-leakage table reports a grid of thresholds with controls. The split threshold came from a rule fixed in the code before the run: it produced a degenerate grouping (one chained giant group), so the script fell back to the fixed value 0.90 (the table "Building the clean split" shows every attempt). The fallback value is a conventional choice, not data-derived.
- **Extras.** A file-name cross-check of the leakage (assumes files with the same UUID prefix are copies of one source photo; not verified against the dataset's documentation), the detector evaluation, and a background-only classifier on image corners.

### What the numbers show

- **Validation accuracy is saturated.** All the base configurations (both backbones, both splits) score above 0.99, so differences between models are small in absolute terms (exact values and seed ranges above). EfficientNet-B3 is above MobileNetV2 by more than the seed ranges, at several times the size and lower throughput.
- **The "clean" split is not clean.** The file-name cross-check finds that a majority of the original validation images have a same-source image in train, and that a large share of the "clean" validation images still do. The embedding method reported far fewer near-copies than the file names indicate: the diagnostic table shows it matched exact copies and some flips well but almost never recognized 90/270-degree, 30-degree or pixel-shifted augmentations. The embedding count is therefore only a lower bound, and the shortfall is large.
- **Consequently the original-vs-clean comparison says little.** Accuracy on the two splits is about the same, which is what a split that barely removes leakage would produce. It must not be read as "leakage does not matter", nor as evidence that the classifier generalizes.
- **Ablations (single seed each).** Freezing the backbone for all epochs cost the most, then the 128 px input and the constant fine-tuning rate; the 160 px input was also below the baseline seed range but by a much smaller margin; a step schedule was indistinguishable from cosine (exact differences in the Observations list above). **Removing augmentation scored higher than the baseline, outside its seed range.** I do not read that as "augmentation does not help": with validation images that have near-identical siblings in training, a model that fits the training images more closely can score higher. That explanation is consistent with the data but untested here.
- **Calibration.** ECE is small on this validation set, but the set is easy and leaky, so it says little about confidence on new images. The app's probabilities remain uncalibrated raw scores.
- **Detector.** The generic COCO "potted plant" detector fired on almost none of the images, so the two-stage pipeline here is the whole-image classifier in practice. This says nothing about a leaf-specific detector or about field photos.
- **Background-only check.** A small MLP on the colour statistics of four 16x16 corners is far above chance. Because corner patches of sibling copies are nearly identical and siblings straddle both splits, this mixes "background carries label information" with "the model recognizes the same photo". The check cannot separate the two on these splits.

### What the numbers do not show

- They do not confirm or refute the 96.7% on my CV: that figure came from a project whose files are lost, with a split, protocol and possible leakage I cannot reconstruct. The numbers here are not comparable with it, and I make no claim about the original.
- They do not show generalization to unseen plants, other cameras, field backgrounds, or partial leaves.
- They are not statistically tested; three seeds give a spread, not a significance test, and ablations are single runs.
- They do not explain *why* an ablation changed anything; the notes above are hypotheses.

### Limits

- **Dataset provenance.** The Kaggle dataset is a third-party augmented derivative of PlantVillage. The original repository's README says that its own Hugging Face train/test splits follow a leaf-grouping logic meant to prevent leakage, and it ships `leaf_grouping/` metadata; I did not use that grouping (the Kaggle file names carry no leaf id), so even a split grouped by file-name UUID would not separate different photos of the same physical leaf.
- **Leakage ground truth is itself an assumption.** The UUID prefix is read as a source-photo id; files without a parsable id (the count is in the table above) are counted as not leaking, so the reported shares are, if anything, underestimates.
- **Embedding split.** The clean split removes only the near-copies the embedding recognized at the chosen fixed threshold; its construction, not the models, is what limits the conclusions. The derived-threshold rule I wrote in advance also failed on real data: grouping at the threshold it produced chained into one giant group, and the similarity distribution is a continuum without two clear modes.
- **Detector.** Generic COCO model restricted to "potted plant"; no leaf-specific detector was used, and no detector was trained (no box labels).
- **Seeds, tests, hardware.** Few seeds; one GPU used on a Kaggle T4 x2 machine; forward-only throughput excludes data loading; no held-out test set.
- **Skipped or not done.** All 20 planned stages ran. Not done: YOLO fine-tuning, probability calibration, field-image evaluation, a source-grouped or leaf-grouped re-split and re-run.

### What I would do next

1. Rebuild the split by source id (and by leaf using the original dataset's grouping metadata), re-run both backbones, and compare against the numbers here. The headline accuracies may move; I have not measured by how much.
2. Evaluate on field-condition images (for example PlantDoc-style data) to measure the lab-to-field gap.
3. Recalibrate probabilities (temperature scaling) on a clean validation set before showing percentages to users.
4. Replace the generic detector with a leaf detector whose licence I can confirm, fine-tuned on labelled boxes, and measure it against the whole-image baseline.

## Licenses and third-party models

- Ultralytics YOLO is licensed AGPL-3.0; the YOLO11n weights are from the official Ultralytics release (file hash and package version are in `results/step5_yolo_eval.json`).
- **Dataset:** ["New Plant Diseases Dataset"](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle (owner: vipoooool). The Kaggle licence field reads "Data files © Original Authors". The original source is [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (Mohanty, Hughes and Salathé, 2016, DOI 10.3389/fpls.2016.01419); I checked that repository on 2026-10-15 and it states no licence (see LICENSES.md). **I use this dataset only for my own learning and portfolio work and do not redistribute it.** No images, galleries or weights trained on it are committed. Source credited; original authors' terms unconfirmed. Details: [LICENSES.md](LICENSES.md).
- torchvision ImageNet weights (MobileNetV2, EfficientNet-B3, ResNet-50) are downloaded at run time.
- The licence of this repo's own code is not chosen yet; a recommendation is in DECISIONS.md.
