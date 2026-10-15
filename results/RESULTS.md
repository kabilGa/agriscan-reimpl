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
