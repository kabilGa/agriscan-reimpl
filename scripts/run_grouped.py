"""Kaggle runner for the second protocol (grouped splits).

1. rebuild the photo- and leaf-grouped splits from the dataset folder and check them against splits/split_hashes.json,
2. train MobileNetV2 and EfficientNet-B3, ONE seed each, on the leaf+photo-grouped split (primary) and then on the
   photo-grouped split (intermediate). The held-out test part is evaluated once per run, after its last epoch.
A stage whose result file exists is skipped, so re-running after a crash continues where it stopped.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
sys.path.insert(0, str(REPO / "src"))
from agriscan.data import find_dataset_root  # noqa: E402
from agriscan.utils import env_info  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="auto", help="dataset root, or 'auto' to search /kaggle/input")
ap.add_argument("--workdir", default=None, help="where results/, splits/, artifacts/, export/ are written (default: the repo)")
ap.add_argument("--preset", default="full", choices=["full", "smoke"])
ap.add_argument("--leaf-map", default="artifacts/leaf-map.json", help="cache path of the authors' leaf map (downloaded if missing)")
ap.add_argument("--expected", default=str(REPO / "splits" / "split_hashes.json"))
ap.add_argument("--allow-mismatch", action="store_true", help="continue even if the rebuilt splits differ from the committed hashes")
ap.add_argument("--max-hours", type=float, default=2.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

WORK = Path(args.workdir).resolve() if args.workdir else REPO
WORK.mkdir(parents=True, exist_ok=True)
SMOKE = args.preset == "smoke"
S = lambda name: str(REPO / "scripts" / name)  # noqa: E731
T = (["--no-pretrained", "--img-size", "64", "--epochs-head", "1", "--epochs-ft", "1", "--batch", "16", "--workers", "0",
      "--limit-train", "60", "--limit-val", "20", "--limit-test", "20"] if SMOKE else ["--workers", "4"])
EST_MIN = {"mobilenet_v2": 20, "efficientnet_b3": 35}  # rough minutes, used only by the budget check

if SMOKE and args.data == "auto":
    raise SystemExit("--preset smoke needs --data <toy dataset folder>")
root = args.data if SMOKE or args.data != "auto" else str(find_dataset_root())
print("dataset root:", root)


def train(arch, split, save=False):
    name = f"{arch}__{split}__s{args.seed}__base"
    cmd = [PY, S("04_train.py"), "--arch", arch, "--split", split, "--root", root, "--seed", str(args.seed), "--tag", "base",
           "--role", "core", *T] + (["--save-model"] if save else [])
    return {"name": name, "cmd": cmd, "arch": arch, "done": [f"results/runs/{name}.json"]}


stages = [{"name": "build_grouped_splits", "arch": None,
           "cmd": [PY, S("11_make_grouped_splits.py"), "--root", root, "--leaf-map", args.leaf_map, "--expected", args.expected,
                   "--out", "splits", "--results", "results/step8_grouped_splits.json", "--seed", str(args.seed)]
           + (["--allow-mismatch"] if args.allow_mismatch else []),
           "done": ["results/step8_grouped_splits.json", "splits/photo_split.csv.gz", "splits/leaf_split.csv.gz"]},
          train("mobilenet_v2", "leaf", save=True), train("efficientnet_b3", "leaf"),
          train("mobilenet_v2", "photo"), train("efficientnet_b3", "photo")]
if args.dry_run:
    for s in stages:
        print(s["name"])
    raise SystemExit

status_path = WORK / "results" / "run_grouped_status.json"
status = {"preset": args.preset, "env": env_info(), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": []}
t_start = time.time()
measured = {}


def record(name, state, why="", minutes=0.0):
    status["stages"].append({"name": name, "state": state, "why": why, "minutes": round(minutes, 1)})
    status["elapsed_min"] = (time.time() - t_start) / 60
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2))
    print(f"=== {name}: {state} {why} (elapsed {status['elapsed_min']:.1f} min)", flush=True)


for s in stages:
    name = s["name"]
    if all((WORK / d).exists() for d in s["done"]):
        record(name, "done_earlier", "result files exist")
        continue
    est = measured.get(s["arch"]) or EST_MIN.get(s["arch"], 5)
    if not SMOKE and s["arch"] and (time.time() - t_start) / 60 + est > args.max_hours * 60:
        record(name, "skipped", f"time budget: {args.max_hours} h would be exceeded (estimate {est:.0f} min)")
        continue
    print(f"\n>>> {name}: {' '.join(map(str, s['cmd'][1:]))}", flush=True)
    t0 = time.time()
    rc = subprocess.run(s["cmd"], cwd=WORK).returncode
    mins = (time.time() - t0) / 60
    if rc == 0 and all((WORK / d).exists() for d in s["done"]):
        if s["arch"]:
            measured[s["arch"]] = mins
        record(name, "done", minutes=mins)
    else:
        record(name, "FAILED", f"exit code {rc}", mins)
        if s["arch"] is None:  # without the splits nothing else can run
            break

export = WORK / "export"
(export / "models").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(export / "agriscan_grouped_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ("results/step8_grouped_splits.json", "results/run_grouped_status.json", "results/runs/*__photo__*.json",
                "results/runs/*__leaf__*.json", "splits/photo_split.csv.gz", "splits/leaf_split.csv.gz"):
        for p in WORK.glob(pat):
            z.write(p, p.relative_to(WORK))
for p in (WORK / "artifacts/models").glob("*.pt"):
    shutil.copy(p, export / "models" / p.name)  # weights: download them, never commit them
failed = [x["name"] for x in status["stages"] if x["state"] == "FAILED"]
print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min. Failed stages: {failed or 'none'}. Download {export}/agriscan_grouped_results.zip")
sys.exit(1 if failed else 0)
