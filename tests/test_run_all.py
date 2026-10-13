"""The Kaggle 'Run all' orchestrator, end to end in smoke mode on a toy dataset (slow: ~17 tiny trainings)."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from synth import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_autodiscovery_fails_clearly(tmp_path):
    r = subprocess.run([sys.executable, "-c",
                        f"from agriscan.data import find_dataset_root; find_dataset_root(r'{tmp_path}')"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "No plant-disease dataset found" in r.stderr and "Add Input" in r.stderr


def test_run_all_smoke_and_resume(tmp_path):
    data, work = tmp_path / "data", tmp_path / "out"
    make_dataset(data, n_orig=15, size=256)   # 256px like the real images, so the step-1 scripts see realistic input
    cmd = [sys.executable, "scripts/run_all.py", "--preset", "smoke", "--data", str(data), "--workdir", str(work)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    st = json.loads((work / "results" / "run_status.json").read_text())
    states = {s["name"]: s["state"] for s in st["stages"]}
    assert "FAILED" not in states.values(), states
    assert states["step1_clean_split"] == "done" and states["mobilenet_v2__clean__s0__base"] == "done"
    assert states["step5_yolo_eval"] == "skipped"
    assert len(list((work / "results" / "runs").glob("*.json"))) == 15   # 4 core + 6 ablations + 4 seeds + resnet50 extra
    assert (work / "results" / "summary.md").exists() and (work / "export" / "agriscan_results.zip").exists()
    assert (work / "export" / "models" / "mobilenet_v2__clean__s0__base.pt").exists()
    assert not any(n.endswith(".pt") for n in zipfile.ZipFile(work / "export" / "agriscan_results.zip").namelist())
    clean = json.loads((work / "results" / "step1_clean_split.json").read_text())["metrics"]
    assert clean["clean_split"]["of_which_same_label"] == 0

    r2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)   # second run: everything already done
    assert r2.returncode == 0 and "done_earlier" in r2.stdout and ">>> " not in r2.stdout
