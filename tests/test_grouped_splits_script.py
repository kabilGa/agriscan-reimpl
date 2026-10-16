"""scripts/11 end to end on a synthetic dataset: both input modes agree, the hash check passes and fails properly."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from synth import make_grouped_dataset, write_leaf_map

ROOT = Path(__file__).resolve().parents[1]


def _run(*args, check=True):
    r = subprocess.run([sys.executable, "scripts/11_make_grouped_splits.py", *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    if check:
        assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    return r


def test_build_check_and_mismatch(tmp_path):
    data = tmp_path / "data"
    lm = write_leaf_map(make_grouped_dataset(data, n_sources=40), tmp_path / "leaf-map.json")
    out, res, exp = tmp_path / "splits", tmp_path / "res.json", tmp_path / "exp.json"
    # 1) build from the dataset folder (the Kaggle path) and record the hashes
    _run("--root", data, "--leaf-map", lm, "--out", out, "--results", res, "--expected", exp, "--write-expected")
    m = json.loads(res.read_text())["metrics"]
    assert m["photo"]["groups_straddling_parts"] == 0 and m["leaf"]["leaf_ids_straddling_parts"] == 0
    assert m["leaf"]["same_leaf_leakage_of_mapped_files"]["val"]["share_whose_leaf_is_in_train"] == 0.0
    assert m["naming"]["classes_without_uuid"] == {"C___scab": m["naming"]["files_without_uuid"]}
    assert 0.5 < m["leaf_coverage"]["fraction"] < 0.95
    leaf = pd.read_csv(out / "leaf_split.csv.gz")
    assert list(leaf.columns) == ["rel", "label", "split", "group", "leaf_mapped"] and set(leaf.split) == {"train", "val", "test"}
    # 2) rebuild from the committed-style file list: identical hashes -> check passes
    out2 = tmp_path / "splits2"
    r = _run("--files", out / "photo_split.csv.gz", "--leaf-map", lm, "--out", out2, "--results", tmp_path / "r2.json", "--expected", exp)
    assert "MISMATCH" not in r.stdout and r.stdout.count("MATCH") >= 4
    assert (out / "leaf_split.csv.gz").read_bytes() == (out2 / "leaf_split.csv.gz").read_bytes()
    # 3) a different file list (one file removed) must be caught
    f = pd.read_csv(out / "photo_split.csv.gz")[["rel", "label"]].iloc[1:]
    f.to_csv(tmp_path / "fewer.csv", index=False)
    bad = _run("--files", tmp_path / "fewer.csv", "--leaf-map", lm, "--out", tmp_path / "s3", "--results", tmp_path / "r3.json", "--expected", exp, check=False)
    assert bad.returncode == 2 and "MISMATCH" in bad.stdout and "differ" in bad.stderr
    ok = _run("--files", tmp_path / "fewer.csv", "--leaf-map", lm, "--out", tmp_path / "s4", "--results", tmp_path / "r4.json", "--expected", exp, "--allow-mismatch")
    assert "MISMATCH" in ok.stdout
