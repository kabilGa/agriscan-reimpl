# Licences and third-party material

What this repo uses, under which terms, and what is **not** confirmed. Nothing here is legal advice.

## Dataset

| Item | Value |
|---|---|
| Name | New Plant Diseases Dataset |
| Owner on Kaggle | vipoooool |
| URL | https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset |
| Licence field on the Kaggle page | "Data files © Original Authors" |
| "Original source" link on the page | The same Kaggle URL as above (as relayed by the repo owner); it does not identify the underlying authors |
| Recorded on | 2026-10-14, from the Kaggle page as read by the repo owner (my tools could not render that page, so I did not verify it independently) |

**Status: no open licence granted; data not redistributed; source credited; original authors' terms unconfirmed.**

What that means for this repo:

- The images are **not** in this repo and never will be: the dataset is downloaded from Kaggle by each user, and `data/` is git-ignored.
- Anything that would contain dataset images stays out of git too: the pair galleries (`artifacts/leakage/pairs_*.png`) and the `artifacts/` folder.
- Committed derived metadata is limited to file paths relative to the dataset root, class labels, group ids (`splits/clean_split.csv.gz`) and aggregate numbers (`results/*.json`). If the original authors object to even that, remove `splits/`.
- Trained model weights are not committed (also because the licence status of weights trained on this data is unconfirmed).
- Use is limited to personal, non-commercial research and learning. Anyone reusing this repo must check the original authors' terms themselves.
- Third-party mirrors describe the dataset as built by offline augmentation from an original dataset; I have not confirmed who the original authors are, which is why their terms are listed as unconfirmed.

## Models and software

| Component | Licence / terms | How it is used here |
|---|---|---|
| Ultralytics YOLO11n (COCO weights) | AGPL-3.0 (Ultralytics) | Downloaded at run time from the official Ultralytics release; never committed. AGPL can affect how combined code is licensed |
| torchvision ImageNet weights (MobileNetV2, EfficientNet-B3, ResNet-50) | Downloaded from the PyTorch hub at run time; terms as published by PyTorch/torchvision (not audited here) | Not committed, not redistributed |
| PyTorch, torchvision, NumPy, pandas, SciPy, matplotlib, Pillow, FastAPI, uvicorn, httpx | Their own open-source licences | Installed from PyPI; not vendored; not exhaustively audited |

## Considered and not used

- Hugging Face `pedromiguelsanchez/yolo-plant-leaf-detection` and `sanjayBalu/leaf-detection` (leaf detector, tagged MIT): licence chain **unconfirmed** (fine-tune of AGPL-3.0 YOLO11 weights, training data licence not stated, pickle file). Not used. See DECISIONS.md, 5.1.

## This repo's own code

Licence not chosen yet; to be decided with the README review before any publication (note the Ultralytics AGPL-3.0 point above).
