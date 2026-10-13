"""One command for the whole Kaggle run: step 1, then the core runs, ablations, seed replicates, YOLO eval, extras.

Stages run in priority order. A stage whose result file already exists is skipped (so re-running after a crash
continues where it stopped). Before each stage after step 1, a time-budget check skips stages that would not fit;
every decision is recorded in results/run_status.json. This file trains nothing itself: it only calls the scripts.
"""
import argparse
import importlib.util
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
ap.add_argument("--preset", default="full", choices=["full", "smoke"], help="smoke = tiny CPU run for testing the plumbing")
ap.add_argument("--max-hours", type=float, default=10.0, help="do not START a stage that would not fit in this budget")
ap.add_argument("--resume-from", default=None, help="a previous run's output folder to copy results/ckpt/features from")
ap.add_argument("--with-yolo", action="store_true", help="in smoke preset, also run the YOLO stage (needs ultralytics + download)")
ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
args = ap.parse_args()

WORK = Path(args.workdir).resolve() if args.workdir else REPO
WORK.mkdir(parents=True, exist_ok=True)
SMOKE = args.preset == "smoke"
S = lambda name: str(REPO / "scripts" / name)  # noqa: E731  scripts live in the repo, outputs in WORK
T = ["--no-pretrained", "--img-size", "64", "--epochs-head", "1", "--epochs-ft", "1", "--batch", "16", "--workers", "0",
     "--limit-train", "60", "--limit-val", "30"] if SMOKE else ["--workers", "4"]
LEAK = ["--no-pretrained", "--limit", "200", "--workers", "0"] if SMOKE else ["--workers", "4"]
FEATS = "artifacts/leakage/feats_limit200.npz" if SMOKE else "artifacts/leakage/feats.npz"
# Rough minutes per training run by backbone: used ONLY by the budget check until a real run has been timed.
EST_MIN = {"mobilenet_v2": 25, "efficientnet_b3": 70, "resnet50": 45}
MB, EB = "mobilenet_v2", "efficientnet_b3"
MAIN = f"{MB}__clean__s0__base"

if args.resume_from:
    src = Path(args.resume_from)
    for sub in ("results", "splits", "artifacts/leakage", "artifacts/ckpt", "artifacts/models"):
        if (src / sub).exists() and not (WORK / sub).exists():
            shutil.copytree(src / sub, WORK / sub)
            print("resumed", sub, "from", src)

if SMOKE and args.data == "auto":
    raise SystemExit("--preset smoke needs --data <toy dataset folder>")
root = args.data if SMOKE or args.data != "auto" else str(find_dataset_root())
print("dataset root:", root)


def train(arch, split, seed=0, tag="base", role="core", extra=(), save=False):
    name = f"{arch}__{split}__s{seed}__{tag}"
    cmd = [PY, S("04_train.py"), "--arch", arch, "--split", split, "--root", root, "--seed", str(seed), "--tag", tag,
           "--role", role, *extra, *T] + (["--save-model"] if save else [])
    return {"name": name, "cmd": cmd, "arch": arch, "role": role, "done": f"results/runs/{name}.json",
            "needs": ["splits/clean_split.csv.gz"] if split == "clean" else []}


def other(name, cmd, done, role="core", needs=(), skip=None, est_min=15):
    return {"name": name, "cmd": cmd, "arch": None, "role": role, "done": done, "needs": list(needs), "skip": skip, "est_min": est_min}


stages = [
    other("step1_inspect", [PY, S("01_inspect_data.py"), "--root", root], "results/step1_inspect.json"),
    other("step1_leakage", [PY, S("02_leakage.py"), "--root", root, *LEAK], "results/step1_leakage.json"),
    other("step1_clean_split", [PY, S("03_clean_split.py"), "--feats", FEATS, "--threshold", "auto"],
          "results/step1_clean_split.json", needs=["results/step1_leakage.json"]),
    # core: both backbones on both splits (clean first: it is the number to trust)
    train(MB, "clean", save=True), train(MB, "original"), train(EB, "clean", save=True), train(EB, "original"),
    # ablations on the clean split, one factor at a time vs the MobileNetV2 baseline
    train(MB, "clean", tag="frozen_only", role="ablation", extra=["--epochs-head", "6", "--epochs-ft", "0"]),
    train(MB, "clean", tag="no_aug", role="ablation", extra=["--no-aug"]),
    train(MB, "clean", tag="size128", role="ablation", extra=["--img-size", "128"]),
    train(MB, "clean", tag="size160", role="ablation", extra=["--img-size", "160"]),
    train(MB, "clean", tag="sched_constant", role="ablation", extra=["--schedule", "constant"]),
    train(MB, "clean", tag="sched_step", role="ablation", extra=["--schedule", "step"]),
    # seed replicates of the two backbones on the clean split
    train(MB, "clean", seed=1, role="seed_replicate"), train(MB, "clean", seed=2, role="seed_replicate"),
    train(EB, "clean", seed=1, role="seed_replicate"), train(EB, "clean", seed=2, role="seed_replicate"),
    other("step5_yolo_eval", [PY, S("05_yolo_eval.py"), "--model", f"artifacts/models/{MAIN}.pt", "--root", root,
                              *(["--n", "20"] if SMOKE else [])], "results/step5_yolo_eval.json",
          needs=[f"artifacts/models/{MAIN}.pt"], est_min=10,
          skip="skipped in smoke preset (use --with-yolo)" if SMOKE and not args.with_yolo else
          (None if importlib.util.find_spec("ultralytics") else "ultralytics not installed")),
    # extras, last: only if everything above is saved and the budget allows
    other("extra_background_check", [PY, S("07_background_check.py"), "--root", root, "--workers", "0" if SMOKE else "4",
                                     *(["--limit", "100"] if SMOKE else [])], "results/extra_background_check.json",
          role="extra", est_min=8),
    train("resnet50", "clean", role="extra"),
]
if args.dry_run:
    for s in stages:
        print(s["role"].ljust(15), s["name"])
    raise SystemExit

status_path = WORK / "results" / "run_status.json"
status = {"preset": args.preset, "env": env_info(), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": []}
t_start = time.time()
measured = {}  # arch -> minutes of the last finished training stage


def record(name, state, why="", minutes=0.0):
    status["stages"].append({"name": name, "state": state, "why": why, "minutes": round(minutes, 1)})
    status["elapsed_min"] = (time.time() - t_start) / 60
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2))
    print(f"=== {name}: {state} {why} (elapsed {status['elapsed_min']:.1f} min)", flush=True)


for s in stages:
    name = s["name"]
    missing = [n for n in s["needs"] if not (WORK / n).exists()]
    if (WORK / s["done"]).exists():
        record(name, "done_earlier", "result file exists")
    elif missing:
        record(name, "skipped", f"missing prerequisite {missing}")
    elif s.get("skip"):
        record(name, "skipped", s["skip"])
    else:
        est = measured.get(s["arch"]) or EST_MIN.get(s["arch"]) or s["est_min"]
        if not SMOKE and not name.startswith("step1") and (time.time() - t_start) / 60 + est > args.max_hours * 60:
            record(name, "skipped", f"time budget: {args.max_hours} h would be exceeded (estimate {est:.0f} min)")
            continue
        print(f"\n>>> {name}: {' '.join(map(str, s['cmd'][1:]))}", flush=True)
        t0 = time.time()
        rc = subprocess.run(s["cmd"], cwd=WORK).returncode
        mins = (time.time() - t0) / 60
        if rc == 0 and (WORK / s["done"]).exists():
            if s["arch"]:
                measured[s["arch"]] = mins
            record(name, "done", minutes=mins)
        else:
            record(name, "FAILED", f"exit code {rc}", mins)

# Aggregate and export everything that should come back to the laptop.
subprocess.run([PY, S("08_aggregate.py")], cwd=WORK)
export = WORK / "export"
(export / "models").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(export / "agriscan_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ("results/**/*.json", "results/*.md", "splits/*.csv.gz", "artifacts/leakage/pairs_*.png", "artifacts/inspect/*.png"):
        for p in WORK.glob(pat):
            z.write(p, p.relative_to(WORK))
for p in (WORK / "artifacts/models").glob("*.pt"):
    shutil.copy(p, export / "models" / p.name)  # weights: download them, never commit them
failed = [x["name"] for x in status["stages"] if x["state"] == "FAILED"]
print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min. Failed stages: {failed or 'none'}. "
      f"Download {export}/ (results zip + models).")
