# Licences and third-party material

What this repo uses, under which terms, and what is **not** confirmed. Nothing here is legal advice.

## Dataset

**My statement: I use this dataset only for my own learning and portfolio work and do not redistribute it.**

| Item | Value |
|---|---|
| Name | New Plant Diseases Dataset |
| Owner on Kaggle | vipoooool |
| URL | https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset |
| Licence field on the Kaggle page | "Data files © Original Authors" |
| Original source | https://github.com/spMohanty/PlantVillage-Dataset (PlantVillage; S. P. Mohanty, D. P. Hughes, M. Salathé, "Using deep learning for image-based plant disease detection", Frontiers in Plant Science 7, 2016, DOI 10.3389/fpls.2016.01419) |
| Kaggle licence field recorded on | 2026-10-14, as read from the Kaggle page by the repo owner (my tools could not render that page) |
| Original repository checked on | 2026-10-15 (UTC), by me, by downloading its files and its GitHub API metadata |

**Status: no open licence granted; data not redistributed; source credited; original authors' terms unconfirmed.**

### What the original source states (checked 2026-10-15)

- **Licence: none stated.** The repository has no LICENSE, LICENSE.md, LICENSE.txt or COPYING file on its default branch (each returned HTTP 404), the README has no licence section, and GitHub's API reports `license: null` for the repository. I infer no permissions from that.
- **Description:** the README calls the repository an "open access repository" of 54,306 images of healthy and diseased leaves (14 crop species, 26 diseases). That phrase is a description, not a licence grant, and I do not treat it as one.
- **Attribution request:** the README says "please consider citing" the 2016 paper above and gives a BibTeX entry (authors Mohanty, Hughes and Salathé; Digital Epidemiology Lab, EPFL). I follow that request here and in the README.
- **Leakage note:** the README says that the train/test splits it offers through Hugging Face follow a leaf-grouping logic meant to prevent data leakage, and the repository ships `leaf_grouping/` metadata. This repo does not use that grouping (see README, "Limits").
- **Kaggle copy:** the Kaggle dataset is a third-party derivative (augmented). I did not verify the relationship to the original beyond the file naming and the owner's statement; the Kaggle licence field says "Data files © Original Authors".

Because redistribution is not clearly permitted by either source, nothing that contains dataset images is committed.

### What that means for this repo

- The images are **not** in this repo and never will be: the dataset is downloaded from Kaggle by each user, and `data/` is git-ignored.
- Anything that contains dataset images stays out of git: the pair galleries and sample grid (`artifacts/`), `agriscan_results/`, `export/`, and any `*.png` / `*.jpg` (all git-ignored).
- Committed derived metadata is limited to file paths relative to the dataset mount, class labels, split names and group ids (`splits/clean_split.csv.gz`) and aggregate numbers (`results/*.json`). The paths contain the dataset's own file names. If the original authors object to even that, remove `splits/`.
- Trained model weights are not committed (the licence status of weights trained on this data is unconfirmed).
- Use is limited to my own learning and portfolio work. Anyone reusing this repo must check the original authors' terms themselves.

## Models and software

| Component | Licence / terms | How it is used here |
|---|---|---|
| Ultralytics YOLO11n (COCO weights) | AGPL-3.0 (Ultralytics) | Downloaded at run time from the official Ultralytics release; never committed. AGPL can affect how combined code is licensed |
| torchvision ImageNet weights (MobileNetV2, EfficientNet-B3, ResNet-50) | Downloaded from the PyTorch hub at run time; terms as published by PyTorch/torchvision (not audited here) | Not committed, not redistributed |
| PyTorch, torchvision, NumPy, pandas, SciPy, matplotlib, Pillow, FastAPI, uvicorn, httpx | Their own open-source licences | Installed from PyPI; not vendored; not exhaustively audited |

## Considered and not used

- Hugging Face `pedromiguelsanchez/yolo-plant-leaf-detection` and `sanjayBalu/leaf-detection` (leaf detector, tagged MIT): licence chain **unconfirmed** (fine-tune of AGPL-3.0 YOLO11 weights, training data licence not stated, pickle file). Not used. See DECISIONS.md, 5.1.

## This repo's own code

Licence not chosen yet. A recommendation, with the reasoning and the AGPL-3.0 implications of the optional Ultralytics dependency, is in DECISIONS.md (section "Code licence recommendation"); the choice is the owner's.

## The authors' leaf map (used for the leaf-grouped split)

`leaf_grouping/leaf-map.json` from the original PlantVillage repository (file name to leaf id, about 40,000 entries) is used only to decide which photos belong to the same physical leaf. The repository states no licence (see above), so:

- the file is **downloaded at run time** into the git-ignored `artifacts/` folder and is **never committed**;
- its SHA-256 is recorded in `results/step8_grouped_splits.json` and `splits/split_hashes.json`, so a different version of the file is detected;
- the committed split files (`splits/photo_split.csv.gz`, `splits/leaf_split.csv.gz`) contain only dataset file paths, labels, split names, group ids and a boolean `leaf_mapped`; they contain **no leaf ids** and none of the map's content.
