# Decisions

Every choice, why, and the alternatives. Numbered by step.

## Step 0: environment and repo setup

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 0.1 | New standalone repo `agriscan-reimpl`, separate from any other project | Keeps history, secrets and files clean | Subfolder of an existing repo |
| 0.2 | Repo will be public on GitHub, **not pushed until the owner says so** | Kaggle can `git clone` a public repo | Private repo + token (risks secret leaks) |
| 0.3 | `LEARNING_NOTES.md` and `INTERVIEW_QA.md` are git-ignored | Personal study material; must not go public by accident | Commit them (rejected by owner) |
| 0.4 | No local GPU: real training on Kaggle GPU; laptop for code and CPU smoke tests | A full run on a laptop CPU would take days | Colab, paid cloud |
| 0.5 | PyTorch only | One framework is easier to explain; README states the original also used TensorFlow | PyTorch + TensorFlow |
| 0.6 | `requirements.txt` / `pyproject.toml` use lower bounds, not pins; exact versions are saved in each results JSON | Kaggle ships its own torch; pinning would reinstall GBs or break CUDA | `pip freeze` lockfile |
| 0.7 | Every run writes JSON with seed, config, date, library versions, device, git commit (`+dirty` if uncommitted) | Honesty rule: every reported metric traceable to a run | Printing numbers in notebooks |
| 0.8 | Repo-local git identity = GitHub no-reply address | Keeps personal/university email out of public history | Global identity |
| 0.9 | Commits carry `Co-Authored-By: Claude`; README says Claude Code was used | Owner wants to be open about it | Omit |
| 0.10 | Local Python 3.14.7 with CPU torch; `requires-python >= 3.10` | Installed version works; Kaggle is 3.10+ | Older Python |
| 0.11 | Teammate names are NOT in any committed file until their consent is confirmed | Owner's rule (personal information) | Name them now |

## Step 1: data, leakage measurement, clean split

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 1.1 | Scripts *search* for `train`/`valid`/`test`; `find_dataset_root` scans `/kaggle/input/*` and fails with a clear message | Exact Kaggle nesting/slug not verifiable from here (Kaggle page would not render) | Hard-coded slug/path |
| 1.2 | Near-copy detection = cosine similarity of ImageNet MobileNetV2 pooled features | Augmentations (flip/rotate/crop/brightness) break perceptual hashes; CNN features are more tolerant, no extra training | pHash/dHash, SSIM, DINO/CLIP |
| 1.3 | Report BOTH raw (L2-normalized) and mean-centered features; the split uses centered | Post-ReLU features are all positive so raw cosines are compressed upward; showing both lets the reader judge | Only one variant |
| 1.4 | **Threshold rule fixed in code before seeing real data** (`pick_threshold`): final = max(A, B). A = 99.9th percentile of best-match similarity to images of a *different class* (known non-copies). B = valley of the val-to-train similarity histogram above A (dip between "no copy" and "copy" modes). Fixed grid 0.80-0.99 is reported next to it | No ground truth exists for "same photo"; a data-derived rule written in advance avoids choosing the number that gives a nice result | Fixed 0.95 by convention; picking by eye after seeing results |
| 1.5 | Fallback ladder in `03_clean_split.py --threshold auto`: if the derived threshold gives a degenerate split (a group larger than max(1% of images, 50), or any class whose validation share is outside 0.5x-1.5x of 20%), try 0.90, 0.95, 0.98; every attempt is recorded in the JSON, and `split_is_healthy` says whether any worked | Found on the toy data: with no clear valley, rule A alone was too low and glued same-class images into giant groups. An unattended run must not silently produce a broken split. The ladder values are conventional, not tuned | Crash; accept the degenerate split |
| 1.6 | Calibration controls: best match among *other classes*, train-to-train, the 33 test images vs train | Show what unrelated look-alikes score, how duplicated train already is, what non-augmented images look like | Only val-to-train |
| 1.7 | Clean split: link same-label pairs >= threshold, connected components = groups, whole groups to train or val per class (~20% val) | A copy and its sibling never straddle the split; per-class keeps all 38 classes in val | Drop leaky val images (biases val); unstratified group split |
| 1.8 | Cross-label near pairs are counted, not linked; a group that would push a class's val share past 1.25x the target stays in train | Linking would glue classes; the cap stops one chain-linked group flooding val | No cap |
| 1.9 | Pool = original train + valid together, then re-split | Otherwise copies across the old boundary are not seen | Re-split train only |
| 1.10 | Clean split saved as `splits/clean_split.csv.gz` (relative paths, label, split, group; no images); to be committed | Needed so every Kaggle run uses the identical split; metadata, not data | Recompute each time (float noise can shift borderline groups) |
| 1.11 | The 33 test images are used only as a control, not for evaluation | Folder has no class subfolders (labels would need filename parsing); 33 images are too few to be a useful test set | Parse labels from filenames |

**Known limits of the leakage estimate:** it is a *lower bound* (copies the embedding misses, e.g. strong rotation, are not counted); it depends on the threshold (a sensitivity table is reported); look-alike photos that are not augmentations of one file may also match; the optional background-only check is run only if the budget allows.

## Steps 2-4: training, backbones, ablations

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 2.1 | Two stages: (1) 2 epochs, backbone frozen, AdamW lr 1e-3 on the new head; (2) 4 epochs, last blocks unfrozen, AdamW lr 1e-4 with per-epoch cosine decay | The new head starts random: big early gradients would wreck pretrained features if everything trained at once. Epoch counts are small because of GPU quota and were fixed a priori, not tuned on validation | Train all layers from the start; LR range test |
| 2.2 | "Last blocks": MobileNetV2 `features[13:]`, EfficientNet-B3 `features[6:]`, ResNet-50 `layer4` | Roughly the last third of each network: high-level, task-specific features | Unfreeze everything; unfreeze only the last block |
| 2.3 | Frozen layers stay in eval mode (BatchNorm statistics fixed); unit-tested | Otherwise running stats drift although weights are frozen | `requires_grad=False` only |
| 2.4 | Plain cross-entropy, AdamW weight decay 1e-4, no label smoothing/mixup | Easiest to explain and defend | Label smoothing, mixup/cutmix |
| 2.5 | Final-epoch metrics are reported; no best-epoch selection | Picking the best epoch on validation would leak validation into the result | Early stopping on val |
| 2.6 | No hyperparameter tuning on validation; no separate test set | Settings were fixed in advance. Ablations compare against the baseline, which can still flatter the best variant slightly | Hold-out test split |
| 2.7 | Online augmentation (RandomResizedCrop 0.6-1, flips, rotation 20, ColorJitter 0.2) on the training side only | The dataset already contains offline augmentations; online ones add variety without creating new copies across the split | Offline only; stronger policies (RandAugment) |
| 2.8 | Input size from the torchvision weights' metadata: B3 300, MobileNetV2 224, ResNet-50 224; images (256 px) are resized directly to that size | Owner's instruction. Torchvision's resize-then-center-crop recipe is simplified because leaves fill the frame | Always 224 |
| 2.9 | Batch 64 (MobileNetV2, ResNet-50), 32 (B3); fp16 autocast on GPU; one GPU even on T4 x2 | B3 at 300 px is memory-hungry (not testable here: no GPU). DataParallel adds complexity | Larger batch; multi-GPU |
| 2.10 | Saved per run: curves, final accuracy, per-class recall, confusion matrix, macro recall, ECE (15 bins), mean confidence, parameters, size, forward-only img/s (+device), training time | Needed for steps 3 and 7 (calibration, confusion, speed, size) | Accuracy only |
| 2.11 | Checkpoint every epoch (atomic write), resume mid-run; finished runs skipped | Kaggle sessions can die; resumed runs are comparable, not bit-identical | Restart from scratch |
| 3.1 | Both backbones on BOTH splits (seed 0); the two backbones also on the clean split with seeds 1 and 2 (`seed_replicate`) when the budget allows; mean and min-max over n seeds are reported | Owner's request; spread shows which differences are noise | One seed |
| 3.2 | ResNet-50 baseline (clean split only) is labelled `extra` and runs last | Optional reproduction of the early experiment | Both splits |
| 4.1 | Ablations on the clean split with MobileNetV2, seed 0, one factor at a time: `frozen_only` (6 head epochs, no fine-tune), `no_aug`, `size128`, `size160`, `sched_constant`, `sched_step` | One-factor design keeps each comparison readable | Grid search |
| 4.2 | Ablations are single-seed; a difference smaller than the baseline's seed spread must not be called an effect | Honest reading | Claim every difference |
| 4.3 | `frozen_only` also differs in total head-training time and uses a constant lr (head stage is always constant) | Documented confound | Matched compute |

## Kaggle orchestration

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| K.1 | `scripts/run_all.py` runs all stages in priority order: step 1, 4 core runs, ablations, seed replicates, YOLO eval, extras; one notebook just calls it | All logic stays testable in the repo; notebook stays dumb | Logic in notebook cells |
| K.2 | Time budget (default 10 h): a stage is skipped if the elapsed time plus its estimate (measured from earlier runs, else rough defaults) would exceed it; skips are recorded in `results/run_status.json` | Kaggle sessions have a time cap and weekly quota; core results must be saved first | Run everything blindly |
| K.3 | Code delivered by a private Kaggle dataset zip (`make_kaggle_bundle.py`, whitelist + final ban check, `BUNDLE_INFO.json` with commit); GitHub clone left as an alternative cell | Repo is not public/pushed yet | Clone from GitHub |
| K.4 | `--preset smoke` runs the whole pipeline on tiny random-init data in a temp folder; part of the test suite | Verifies orchestration, resume and export without a GPU or the dataset | Manual checks |
| K.5 | Results zip contains JSON, the split and gallery PNGs only; model weights are exported separately | Weights must never enter git | One zip |

## Step 5: YOLO stage

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 5.1 | **Considered, not used by default:** HF repos `pedromiguelsanchez/yolo-plant-leaf-detection` and `sanjayBalu/leaf-detection` (one class "Leaf", `yolo11x_leaf.pt`, tagged MIT). Both list the same file and total size (113,925,667 bytes), so one looks like a copy of the other | Licence cannot be confirmed: they are fine-tunes of Ultralytics YOLO11 (AGPL-3.0) labelled MIT; the training data (Kaggle "leaf-detection" by `alexo98`) has no stated licence; the `.pt` is a pickle, so loading it executes code from an unvetted uploader. Owner's rule: record and skip that part | Use it anyway |
| 5.2 | Use the **generic official Ultralytics YOLO11n (COCO)**, downloaded at run time by the `ultralytics` package; AGPL-3.0; detections limited to COCO class 58 "potted plant"; confidence threshold 0.25; weights file SHA-256 and package version recorded in the JSON | Verified licence and source; the stage demonstrates the pipeline and is **not a trained component** | Fine-tune a detector (no box labels exist; not requested) |
| 5.3 | No detection => classify the whole image (recorded as fallback); `05_yolo_eval.py` measures on the clean val sample how often nothing is found and compares whole-image vs crop accuracy on the same images | The only honest way to say whether the detector helps. No improvement is claimed unless this shows it | Assume it helps |
| 5.4 | `YoloDetector` takes any Ultralytics weights path (`--detector`), but nothing with an unverified licence is wired in by default | Lets the owner try the leaf detector after reviewing its licence | Hard-code one model |

## Step 6: web app

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 6.1 | Model path only from env `AGRISCAN_MODEL`; the server starts without weights, `/health` says so, `/predict` answers 503 with a clear message; a bad path also gives 503, not a crash | Weights are never in the repo | Crash at start |
| 6.2 | Endpoints: `GET /`, `GET /health`, `POST /predict` (photo, `use_detector` optional), `POST /predict_frame` (video frame) | Owner's spec; minimal | More endpoints |
| 6.3 | Uploads are never stored; 10 MB and 25-megapixel limits; images thumbnailed to 1600 px; non-images give 400, oversize 413; checkpoints are loaded with `weights_only=True` | Basic abuse and unpickling protection for a demo | No limits |
| 6.4 | Every response and the page carry the "demo, lab-style images, not a diagnosis, uncalibrated probabilities" note; the page writes results with `textContent` | Owner's requirement; avoids HTML injection | innerHTML |

## Before-push checklist (nothing is pushed until the owner says so)

- [ ] Teammates have agreed to be named (names are NOT in any committed file now). **Commit `c51e47b` still contains the three names in README/DECISIONS in local history: scrub it (rebase) before any push.**
- [ ] Dataset licence read on the Kaggle page and recorded in the README (could not be verified from here).
- [ ] Repo licence chosen (Ultralytics AGPL-3.0 can affect how combined code may be licensed).
- [ ] `git log --format='%an %ae'` shows only the no-reply address.
- [ ] `git ls-files` has no data, weights, `kaggle.json`, `LEARNING_NOTES.md` or `INTERVIEW_QA.md`; quick grep for tokens/keys.
- [ ] Every README number traces to a JSON in `results/`; all TBDs filled or removed; README reviewed by the owner.

## Known limits

- `set_seed` makes runs comparable, not bit-identical on GPU; resumed runs likewise.
- Nothing has run on the real dataset or a GPU yet; the Kaggle notebook has not been run.
