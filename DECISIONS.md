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
| 0.11 | Teammate names were kept out of committed files until consent; on 2026-10-14 the owner confirmed all three teammates agreed, and they are credited in the README (commit "Docs: credits, dataset licence") | Owner's rule (personal information) | Name them without consent |
| 0.12 | The git history was **not** scrubbed: the names in early commit `c51e47b` are acceptable now that the teammates agreed | Owner's decision, 2026-10-14: scrub skipped | Rebase to remove the names from history |
| 0.13 | Dataset licence handling is recorded in `LICENSES.md`: no open licence granted, data not redistributed, source credited, original authors' terms unconfirmed. No images, galleries or weights are committed; only paths/labels/group ids and aggregate numbers | The Kaggle licence field says "Data files © Original Authors", which is not an open licence | Treat the data as freely redistributable (not justified) |

## Step 1: data, leakage measurement, clean split

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 1.1 | Scripts *search* for `train`/`valid`/`test`; `find_dataset_root` scans `/kaggle/input/*` and fails with a clear message | Exact Kaggle nesting/slug not verifiable from here (Kaggle page would not render) | Hard-coded slug/path |
| 1.2 | Near-copy detection = cosine similarity of ImageNet MobileNetV2 pooled features | Augmentations (flip/rotate/crop/brightness) break perceptual hashes; CNN features are more tolerant, no extra training | pHash/dHash, SSIM, DINO/CLIP |
| 1.3 | Report BOTH raw (L2-normalized) and mean-centered features; the split uses centered | Post-ReLU features are all positive so raw cosines are compressed upward; showing both lets the reader judge | Only one variant |
| 1.4 | **Threshold rule fixed in code before seeing real data** (`pick_threshold`): final = max(A, B). A = 99.9th percentile of best-match similarity to images of a *different class* (known non-copies). B = valley of the val-to-train similarity histogram above A (dip between "no copy" and "copy" modes). Reported next to it: a fixed grid 0.80-0.99 and the thresholds at the 99 / 99.9 / 99.99th percentile of that same non-copy distribution | No ground truth exists for "same photo"; a data-derived rule written in advance avoids choosing the number that gives a nice result | Fixed 0.95 by convention; picking by eye after seeing results |
| 1.5 | Fallback ladder in `03_clean_split.py --threshold auto`: if the derived threshold gives a degenerate split (a group larger than max(1% of images, 50), or any class whose validation share is outside 0.5x-1.5x of 20%), try 0.90, 0.95, 0.98; every attempt is recorded in the JSON, and `split_is_healthy` says whether any worked | Found on the toy data: with no clear valley, rule A alone was too low and glued same-class images into giant groups. An unattended run must not silently produce a broken split. The ladder values are conventional, not tuned | Crash; accept the degenerate split |
| 1.6 | Calibration controls: best match among *other classes*, train-to-train, the 33 test images vs train | Show what unrelated look-alikes score, how duplicated train already is, what non-augmented images look like | Only val-to-train |
| 1.7 | Clean split: link same-label pairs >= threshold, connected components = groups, whole groups to train or val per class (~20% val) | A copy and its sibling never straddle the split; per-class keeps all 38 classes in val | Drop leaky val images (biases val); unstratified group split |
| 1.8 | Cross-label near pairs are counted, not linked; a group that would push a class's val share past 1.25x the target stays in train | Linking would glue classes; the cap stops one chain-linked group flooding val | No cap |
| 1.9 | Pool = original train + valid together, then re-split | Otherwise copies across the old boundary are not seen | Re-split train only |
| 1.10 | Clean split saved as `splits/clean_split.csv.gz` (relative paths, label, split, group; no images); to be committed | Needed so every Kaggle run uses the identical split; metadata, not data | Recompute each time (float noise can shift borderline groups) |
| 1.12 | If a candidate threshold produces more than 50M near pairs (far too low), that attempt is recorded as failed and the ladder moves on; results JSON also stores torch/torchvision/numpy/Pillow/ultralytics/CUDA versions | Unattended robustness; owner asked for library versions | Crash |
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
| 5.5 | `YoloDetector` switches off Ultralytics' anonymous usage analytics (`settings sync=False`) | Privacy-preserving default; no data should leave the machine for a student project | Leave the default (analytics on) |
| 5.6 | Observed locally (provenance, not a result): Ultralytics 8.4.177 fetched `yolo11n.pt` (5.4 MB) from `github.com/ultralytics/assets/releases/download/v8.4.0/`, SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`; COCO class 58 = "potted plant". The Kaggle run records its own version and hash in `results/step5_yolo_eval.json` | Exact source/version requirement | None |
| 5.4 | `YoloDetector` takes any Ultralytics weights path (`--detector`), but nothing with an unverified licence is wired in by default | Lets the owner try the leaf detector after reviewing its licence | Hard-code one model |

## Step 6: web app

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 6.1 | Model path only from env `AGRISCAN_MODEL`; the server starts without weights, `/health` says so, `/predict` answers 503 with a clear message; a bad path also gives 503, not a crash | Weights are never in the repo | Crash at start |
| 6.2 | Endpoints: `GET /`, `GET /health`, `POST /predict` (photo, `use_detector` optional), `POST /predict_frame` (video frame) | Owner's spec; minimal | More endpoints |
| 6.3 | Uploads are never stored; 10 MB and 25-megapixel limits; images thumbnailed to 1600 px; non-images give 400, oversize 413; checkpoints are loaded with `weights_only=True` | Basic abuse and unpickling protection for a demo | No limits |
| 6.4 | Every response and the page carry the "demo, lab-style images, not a diagnosis, uncalibrated probabilities" note; the page writes results with `textContent` | Owner's requirement; avoids HTML injection | innerHTML |

## Phase C: evaluation decisions (after the first real run, 2026-10-15)

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| C.1 | The existing "clean" split is kept and reported as **not clean**, instead of being silently replaced | The file-name cross-check (script 09) shows that most original validation images, and a large share of the clean split's, have a same-source image in train. A replacement needs a new GPU run, which is the owner's call (quota, time) | Swap in a source-grouped split without a re-run (would leave every reported accuracy unexplained) |
| C.2 | Added an extra, labelled analysis: file-name cross-check (`agriscan.crosscheck`, `scripts/09_filename_crosscheck.py`), using only the committed split CSV | It provides a ground-truth-like check that the embedding method lacked. It rests on an unverified assumption (UUID prefix = source photo), which is stated everywhere it is used | Skip it |
| C.3 | All README tables and the "Observations" list are generated by `scripts/10_results_table.py` from the JSON files; a test recomputes cells independently | Rule: no hand-typed numbers. Prose uses design facts (epochs, seeds, thresholds) and refers to the tables for results | Hand-typed tables |
| C.4 | Observations use neutral relative wording ("inside / outside the baseline seed range") and the prose gives hypotheses only as hypotheses (e.g. the no-augmentation run scoring higher) | A single seed cannot support a cause | State causes |
| C.5 | The 96.7% on the CV is neither confirmed nor refuted; the README says the numbers are not comparable | Original files, split and protocol are lost | Compare directly |
| C.6 | Nothing that contains dataset images is committed (galleries, sample grid, `agriscan_results/`, `export/`, `*.png`/`*.jpg` are git-ignored); the original source (PlantVillage repo) states no licence, so redistribution is not shown to be permitted | Consistent with LICENSES.md | Commit galleries |
| C.7 | `.gitignore` rule `runs/` was anchored to `/runs/` | The old rule also hid `results/runs/`, i.e. all per-run JSON files (found by previewing the commit) | Rename the folder |
| C.8 | The Kaggle notebook's `find_code` searches up to 3 folder levels (matches what the run used) | Inputs were mounted under `/kaggle/input/datasets/<owner>/<name>/` | Keep the 2-level glob |

**Findings that matter for how to read the results** (numbers are in the generated tables):
the derived threshold rule produced a chained giant group on real data, so the 0.90 ladder value was used; the embedding grouping recognized exact copies and some flips but almost no 90/270-degree, 30-degree or pixel-shifted copies; validation accuracy is saturated for all base runs; the no-augmentation ablation scored higher than the baseline; the generic detector fired on almost no image; the corner-only classifier is far above chance but is confounded by sibling copies.

## Code licence recommendation (not applied; the choice is the owner's)

**Recommendation: MIT for this repo's own code, with the Ultralytics dependency kept optional and clearly flagged.** If the owner prefers zero ambiguity and might host the app publicly with the detector enabled, choose AGPL-3.0 instead. Not legal advice.

What the facts are:

- Ultralytics (YOLO11) is AGPL-3.0 (an enterprise licence is the alternative). Here it is an **optional extra** (`pip install -e ".[detect]"`), imported lazily in `agriscan/pipeline.py` and used by `scripts/05_yolo_eval.py`, `scripts/06_pipeline_demo.py` and the app only when `AGRISCAN_DETECTOR` is set. No Ultralytics code or weights are in the repo.
- AGPL-3.0 is copyleft and, unlike GPL, also covers use over a network (section 13): whoever runs a modified or combined work as a service for others must offer its complete source under AGPL. Whether `import ultralytics` makes our code a combined work is argued both ways; Ultralytics' own position is that it does.
- MIT code can be combined into an AGPL work, so choosing MIT cannot put the repo in conflict with Ultralytics. The AGPL obligation would fall on whoever distributes or serves the combination with the detector, and they could do so (MIT allows it).
- The detector stage contributes nothing measurable here (it found a plant in almost no image), so the dependency is cheap to drop later if licensing ever matters.

Options:

| Option | Pros | Cons |
|---|---|---|
| **MIT** (recommended) | Short, widely understood, easy reuse, AGPL-compatible; consistent with PyTorch-style ecosystem | Looks permissive while the optional detector path brings AGPL duties for deployers; needs the README/LICENSES note (already present) |
| Apache-2.0 | Same, plus an explicit patent grant and NOTICE requirement | Slightly heavier; patents are not a concern for this project |
| AGPL-3.0 | No ambiguity with Ultralytics; fits if the YOLO stage is central and the app will be hosted | Strong copyleft deters reuse; later closed-source use of this code would need relicensing (possible only for code you alone own) |
| Remove Ultralytics, then MIT/Apache | No AGPL anywhere | Loses the YOLO stage the project is meant to show |

Whichever is chosen: put the owner's name and year in a `LICENSE` file; state that the licence covers the code only, not the dataset (restricted, see LICENSES.md) or any weights; and confirm that none of the code is copied from the lost team project, or that the teammates agree to the licence (this repo is a re-implementation, but the teammates co-own the original).

## Before-push checklist (nothing is pushed until the owner says so)

- [x] Teammates agreed to be named (confirmed by the owner, 2026-10-14); credited in the README. History scrub not needed (owner's decision).
- [x] Dataset licence field and the original source's (absent) licence recorded in README and LICENSES.md.
- [ ] Repo licence chosen (recommendation above; the owner decides, then add `LICENSE` and mention it in the README).
- [ ] `git log --format='%an %ae'` shows only the no-reply address.
- [ ] `git ls-files` has no data, weights, `kaggle.json`, `LEARNING_NOTES.md`, `INTERVIEW_QA.md`, images or galleries; quick grep for tokens/keys.
- [x] Every README number traces to a JSON in `results/` (tables generated by script 10); no TBDs left.
- [ ] README reviewed by the owner.
- [ ] Decide whether to re-run the core experiments on a source-grouped split before publishing the headline accuracies (see README, "What I would do next").

## Known limits

- `set_seed` makes runs comparable, not bit-identical on GPU; resumed runs likewise.
- The first real run (2026-10-15, one T4) is a single execution; the original split and every ablation have one seed each.
- The clean split is clean only against embedding similarity at its threshold, not against shared source photos (see README, "Evaluation and limits").
