"""Generate the Kaggle notebooks from one source, so they cannot drift apart.

  python scripts/make_notebooks.py            # writes notebooks/kaggle_run_all.ipynb and notebooks/kaggle_run_grouped.ipynb
The notebooks only find the code, install it, self-test it and call one script; all logic lives in the repo.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n").splitlines(True)}


GPU = code("""
!nvidia-smi
import torch, sys
print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'No GPU: turn the accelerator on (right sidebar) and restart.'
""")
FIND_CODE = code("""
import glob, os, shutil, subprocess, zipfile
WORK = '/kaggle/working/agriscan'
GITHUB_URL = None   # alternative to the bundle once the repo is public, e.g. 'https://github.com/<user>/agriscan-reimpl'

def find_code():
    # An input can be mounted several folders deep (e.g. /kaggle/input/datasets/<owner>/<name>/), so look up to 3 levels down.
    for depth in (1, 2, 3):
        stars = '/'.join(['*'] * depth)
        for p in glob.glob(f'/kaggle/input/{stars}/pyproject.toml'):
            if 'name = "agriscan"' in open(p).read():
                return os.path.dirname(p)
        for z in glob.glob(f'/kaggle/input/{stars}/*.zip'):
            if 'agriscan' in os.path.basename(z):
                zipfile.ZipFile(z).extractall('/kaggle/working/_unzipped')
                return '/kaggle/working/_unzipped'
    return None

src = find_code()
if src:
    shutil.copytree(src, WORK, dirs_exist_ok=True)
    print('code copied from', src)
elif GITHUB_URL:
    subprocess.run(['git', 'clone', GITHUB_URL, WORK], check=True)
else:
    raise SystemExit('Code not found: add the private code-bundle dataset as an input (see KAGGLE_RUN.md), or set GITHUB_URL.')
print(open(WORK + '/BUNDLE_INFO.json').read() if os.path.exists(WORK + '/BUNDLE_INFO.json') else 'git clone, no BUNDLE_INFO.json')
""")
INSTALL = code("""
# Kaggle ships torch/torchvision; this adds the rest (ultralytics = YOLO stage, pytest = quick self-test).
!pip install -q -e "/kaggle/working/agriscan[dev,app,detect]"
""")
SELFTEST_ALL = code("""
# 1-2 minutes on CPU: logic tests on synthetic data. A failure here means do not trust the run below.
!cd /kaggle/working/agriscan && python -m pytest -q -x tests
""")
SELFTEST = code("""
# 3-4 minutes on CPU: logic tests on synthetic data. A failure here means do not trust the run below.
!cd /kaggle/working/agriscan && python -m pytest -q -x tests
""")

RUN_ALL = [
    md("""
# AgriScan re-implementation: Kaggle "Run all"

**Before running** (right sidebar): Accelerator = **GPU T4 x2 (or P100)**, Internet = **On**, and add two inputs:
1. your private dataset with the code bundle (`agriscan_code.zip`, made by `scripts/make_kaggle_bundle.py`);
2. the public dataset **New Plant Diseases Dataset** (the code finds it by itself; no slug is hard-coded).

Then **Save Version -> Save & Run All (Commit)**, so the run continues if your browser closes.
Everything is written to `/kaggle/working/out`; see `KAGGLE_RUN.md` for what to download. Never paste tokens or passwords here.
"""),
    GPU, FIND_CODE, INSTALL, SELFTEST_ALL,
    code("""
# The whole experiment. Safe to re-run after a crash: finished stages are skipped, unfinished ones resume.
# To continue after the SESSION died: add the previous version's output as an input and set RESUME_FROM to it.
RESUME_FROM = None   # e.g. '/kaggle/input/agriscan-run1/out'
cmd = 'cd /kaggle/working/agriscan && python scripts/run_all.py --data auto --workdir /kaggle/working/out --max-hours 10'
if RESUME_FROM:
    cmd += f' --resume-from {RESUME_FROM}'
!{cmd}
"""),
    code("""
# What to download (Output tab): export/agriscan_results.zip and export/models/*.pt
!ls -la /kaggle/working/out/export /kaggle/working/out/export/models
!cat /kaggle/working/out/results/summary.md
"""),
]

RUN_GROUPED = [
    md("""
# AgriScan re-implementation: second protocol (grouped splits)

Builds a photo-grouped and a leaf+photo-grouped 70/15/15 split from the dataset folder, then trains MobileNetV2 and EfficientNet-B3
(one seed each) on the leaf-grouped split (primary) and the photo-grouped split. About 1.6 h on a T4 (an estimate).

**Before running** (right sidebar): Accelerator = **GPU T4 x2 (or P100)**, **Internet = On** (the authors' leaf map is downloaded at run time),
and add two inputs: the private code-bundle dataset **at its latest version**, and the public **New Plant Diseases Dataset**.
Then **Save Version -> Save & Run All (Commit)**. Everything is written to `/kaggle/working/out2`; see `KAGGLE_RUN.md`.
The split is rebuilt here and compared with the fingerprints committed in the repo: a mismatch stops the run before any training.
Never paste tokens or passwords here.
"""),
    GPU, FIND_CODE, INSTALL, SELFTEST,
    code("""
# Safe to re-run after a crash: finished stages are skipped.
!cd /kaggle/working/agriscan && python scripts/run_grouped.py --data auto --workdir /kaggle/working/out2 --max-hours 2.5
"""),
    code("""
# What to download (Output tab): export/agriscan_grouped_results.zip (and export/models/*.pt if you want the leaf-split checkpoint)
!ls -la /kaggle/working/out2/export /kaggle/working/out2/export/models
!cat /kaggle/working/out2/results/run_grouped_status.json
"""),
]


def write(cells, name):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = NB_DIR / name
    with open(path, "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    write(RUN_ALL, "kaggle_run_all.ipynb")
    write(RUN_GROUPED, "kaggle_run_grouped.ipynb")
