# Decisions

Every choice, why, and the alternatives. Newest steps at the bottom.

## Step 0: environment and repo setup

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 0.1 | New standalone repo `agriscan-reimpl`, separate from any other project | Keeps history, secrets and files clean | Subfolder of an existing repo |
| 0.2 | Repo will be public on GitHub, **not pushed until the owner says so** | Kaggle can `git clone` a public repo; no zip upload | Private repo + token (risks secret leaks), zip upload (manual, error-prone) |
| 0.3 | `LEARNING_NOTES.md` and `INTERVIEW_QA.md` are git-ignored | Personal study material; must not go public by accident | Commit them (rejected by owner) |
| 0.4 | No local GPU (Intel Iris Xe only): real training on Kaggle GPU; laptop for code and CPU smoke tests | A full run on an i7-1355U CPU would take days | Colab (needs a `kaggle.json` upload for the dataset), paid cloud |
| 0.5 | PyTorch only | Owner's choice; one framework is easier to explain | PyTorch + TensorFlow as in the original (README states the original used both) |
| 0.6 | `requirements.txt` uses lower bounds, not exact pins; exact versions are saved in each results JSON | Kaggle/Colab ship their own torch; forcing a pin would reinstall ~GBs or break CUDA | `pip freeze` lockfile (breaks across platforms) |
| 0.7 | Every run writes JSON with seed, config, date, library versions, device, git commit via `save_results()` | Honesty rule: every reported metric must be traceable to a run | Printing numbers in notebooks (not reproducible, easy to misquote) |
| 0.8 | Smoke test uses random data and `weights=None` | Runs offline on CPU in seconds; makes no accuracy claim | Tiny real-data subset (added in step 1 once data is downloaded) |
| 0.9 | Local Python is 3.14.7; CPU torch 2.14.1 installed and verified to run | It is what is installed; `requires-python >= 3.10` keeps Kaggle (3.10+) compatible | Install an older Python (unnecessary, the smoke test passes) |
| 0.10 | Commits carry `Co-Authored-By: Claude`; README says Claude Code was used as a tutor | Owner wants to be open about it | Omit (rejected by owner) |
| 0.11 | Repo-local git identity = GitHub no-reply address; step 0 commit re-authored (nothing was pushed) | Keeps the personal/university email out of public history | Global identity (rejected) |

## Before-push checklist (nothing is pushed until the owner says so)

- [ ] Teammates (Yassine Laabidi, Wassim Ferchichi, Mohamed Fedi Thabet) have agreed to be named in the README; otherwise remove the names.
- [ ] Dataset license read on the Kaggle page and recorded in the README (could not be verified from here).
- [ ] YOLO weights: source/version/license recorded in the README (step 5).
- [ ] Repo license chosen (note: using Ultralytics under AGPL-3.0 can affect how the combined code may be licensed).
- [ ] `git log --format='%an %ae'` shows only the no-reply address.
- [ ] `git ls-files` contains no data, weights, `kaggle.json`, `LEARNING_NOTES.md` or `INTERVIEW_QA.md`; quick grep for tokens/keys.
- [ ] Every README number traces to a JSON in `results/`; all TBDs filled or removed.

## Step 1: data, leakage measurement, clean split

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 1.1 | Scripts *search* for `train`/`valid`/`test` under the given root | Exact Kaggle nesting was not verifiable from here (the Kaggle page would not render; mirrors disagree on folder names) | Hard-code a path (breaks silently if wrong) |
| 1.2 | Near-copy detection = cosine similarity of ImageNet MobileNetV2 pooled features | Augmentations (flip/rotate/crop/brightness) break perceptual hashes; CNN features are more tolerant. Needs no extra training | pHash/dHash (cheap, fragile to rotation/flip); SSIM/pixel distance (fragile); DINO/CLIP embeddings (stronger, heavier, new dependency) |
| 1.3 | Center features (subtract train+val mean) before L2-normalizing | Post-ReLU features are all positive, so raw cosines of unrelated images are already high; centering spreads them out so a threshold is readable | Raw cosine |
| 1.4 | Threshold is NOT fixed in code: the script produces pair galleries per similarity band and the owner chooses by eye; 0.95 is only a provisional default | There is no ground truth for "same photo"; choosing a number silently would be a hidden assumption | Pick 0.95 by convention |
| 1.5 | Calibration controls: best match among *other classes*, train-to-train, and the 33 test images vs train | They show what "unrelated but similar-looking" scores, how duplicated train already is, and what un-augmented images look like | Report only val-to-train |
| 1.6 | Clean split: connect same-label pairs >= threshold, take connected components as groups, assign whole groups to train or val per class (~20% val) | A near-copy and its sibling can never straddle the split. Per-class keeps all 38 classes in val | Drop leaky val images (shrinks/biases val); GroupShuffleSplit without class balancing |
| 1.7 | Cross-label near pairs are counted but not linked | They are label noise or truly confusable classes; linking them would glue classes into giant groups | Link everything |
| 1.8 | A group that would push a class's val share above 1.25x target stays in train | Prevents one chain-linked mega-group from emptying or flooding val for a class; actual per-class fractions are reported | No cap |
| 1.9 | Pool = original train + valid together, then re-split | Clean split must be built over all images, otherwise copies across the old boundary are not seen | Re-split train only |
| 1.10 | Split saved as `splits/clean_split_thr<t>.csv.gz` (relative paths + label + split + group; no images); to be committed | Needed so Kaggle runs use the identical split; it is metadata, not data | Recompute on every run (float noise could shift borderline groups) |
| 1.11 | Tests use synthetic vectors/images and random weights | Verified logic offline without downloading anything; they do not validate the pretrained embedding on real leaves | Download weights locally just for a test |

### Known limits of the leakage estimate

- It is a **lower bound**: copies the embedding fails to match (e.g. strong rotation) are not counted.
- It depends on the chosen threshold; results at several thresholds are reported, not one.
- Same-leaf or same-plant photos that are *not* augmentations of one file may also match; that is arguably still leakage, and the galleries will show it.
- The background-only check (Noyan 2022) is optional and not done yet.

## Known limits of step 0

- `set_seed` makes runs comparable, not bit-identical on GPU (some CUDA kernels are non-deterministic).
- The Kaggle notebook has a `<YOUR_GITHUB_USER>` placeholder until the repo is pushed; it has **not** been run on Kaggle yet.
