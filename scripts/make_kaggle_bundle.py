"""Zip the CODE ONLY, to upload as a PRIVATE Kaggle dataset and attach to the notebook.

A whitelist decides what goes in (source, scripts, tests, notebooks, a few docs). Personal notes, weights,
data, secrets and git history cannot get in: they are not whitelisted, and a final check refuses to write the
zip if any banned name or extension slipped through.
"""
import argparse
import datetime as dt
import json
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIRS = ["src", "scripts", "tests", "notebooks"]
FILES = ["pyproject.toml", "requirements.txt", "README.md", "DECISIONS.md", "KAGGLE_RUN.md", ".gitattributes",
         "splits/split_hashes.json"]  # a few hundred bytes: fingerprints that let Kaggle prove it rebuilt the same split
ALLOWED_EXT = {".py", ".ipynb", ".md", ".toml", ".txt", ".gitattributes", ".json", ""}
BANNED_NAMES = {"LEARNING_NOTES.md", "INTERVIEW_QA.md", "kaggle.json", ".env"}
BANNED_PARTS = {".git", ".venv", "__pycache__", "artifacts", "data", "results", "export", "agriscan.egg-info"}  # egg-info: local build metadata
BANNED_EXT = {".pt", ".pth", ".onnx", ".ckpt", ".safetensors", ".npz", ".gz", ".zip", ".jpg", ".jpeg", ".png", ".h5"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="artifacts/agriscan_code.zip")
args = ap.parse_args()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


files = [REPO / f for f in FILES if (REPO / f).exists()]
for d in DIRS:
    files += [p for p in (REPO / d).rglob("*") if p.is_file()]
files = sorted(p for p in files if p.suffix in ALLOWED_EXT or p.name == ".gitattributes")
files = [p for p in files if not (BANNED_PARTS & set(p.relative_to(REPO).parts[:-1])) and p.name not in BANNED_NAMES]
bad = [p for p in files if p.name in BANNED_NAMES or p.suffix in BANNED_EXT]
assert not bad, f"refusing to bundle: {bad}"

dirty = bool(git("status", "--porcelain"))
info = {"commit": (git("rev-parse", "--short", "HEAD") or "unknown") + ("+dirty" if dirty else ""),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "n_files": len(files)}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(REPO).as_posix())
    z.writestr("BUNDLE_INFO.json", json.dumps(info, indent=2))
print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB, {len(files)} files, code commit {info['commit']})")
if dirty:
    print("WARNING: uncommitted changes in the working tree; commit first so the results name an exact commit.")
