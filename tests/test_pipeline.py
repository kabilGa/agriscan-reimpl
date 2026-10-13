"""End-to-end test of scripts 01-04 on a tiny synthetic dataset in Kaggle's folder layout.

Random embedder/classifier weights are used (no download), so this checks the plumbing only.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from synth import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_scripts_end_to_end(tmp_path):
    data = tmp_path / "data"
    make_dataset(data, size=256)
    art, res = tmp_path / "art", tmp_path / "res"
    _run("scripts/01_inspect_data.py", "--root", data, "--out", art / "inspect", "--results", res / "inspect.json")
    ins = json.loads((res / "inspect.json").read_text())["metrics"]
    assert ins["n_classes"] == 3 and ins["same_classes"]

    _run("scripts/02_leakage.py", "--root", data, "--out", art / "leak", "--results", res / "leak.json",
         "--no-pretrained", "--limit", 200, "--workers", 0)
    leak = json.loads((res / "leak.json").read_text())["metrics"]
    assert leak["centered"]["valid_to_train"]["frac_at_least"]["0.9"] > 0.5   # naive split leaks heavily
    assert set(leak) == {"centered", "raw"}
    thr = leak["centered"]["derived_threshold"]["final"]
    assert 0 < thr < 1

    csv = tmp_path / "split.csv.gz"
    _run("scripts/03_clean_split.py", "--feats", art / "leak" / "feats_limit200.npz", "--threshold", "auto",
         "--leakage-results", res / "leak.json", "--csv", csv, "--results", res / "clean.json")
    clean = json.loads((res / "clean.json").read_text())["metrics"]
    assert clean["clean_split"]["of_which_same_label"] == 0
    assert clean["original_split"]["fraction"] > clean["clean_split"]["val_with_near_copy_in_train"] / clean["clean_split"]["n_val"]
    df = pd.read_csv(csv)
    assert set(df.split) == {"train", "val"}
    assert (df.groupby("group").split.nunique() == 1).all()          # no group straddles the split

    # Step 2-4 entry point on the clean split just produced (tiny random-init net, 1+1 epochs).
    _run("scripts/04_train.py", "--arch", "mobilenet_v2", "--split", "clean", "--root", data, "--csv", csv,
         "--no-pretrained", "--epochs-head", 1, "--epochs-ft", 1, "--img-size", 64, "--workers", 0,
         "--batch", 16, "--results-dir", res / "runs", "--ckpt-dir", art / "ck")
    run = json.loads(next((res / "runs").glob("*.json")).read_text())
    assert run["config"]["split"] == "clean" and len(run["metrics"]["history"]) == 2

    out = _run("scripts/08_aggregate.py", "--runs", res / "runs", "--out", res)
    assert "mobilenet_v2" in out and (res / "summary.json").exists()
