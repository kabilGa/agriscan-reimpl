"""The grouped-split Kaggle runner, end to end in smoke mode on a toy dataset (tiny random-init trainings)."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from synth import make_grouped_dataset, write_leaf_map

ROOT = Path(__file__).resolve().parents[1]


def _cmd(data, work, lm, expected):
    return [sys.executable, "scripts/run_grouped.py", "--preset", "smoke", "--data", str(data), "--workdir", str(work),
            "--leaf-map", str(lm), "--expected", str(expected)]


def test_run_grouped_smoke_resume_and_hash_guard(tmp_path):
    data, work = tmp_path / "data", tmp_path / "out"
    lm = write_leaf_map(make_grouped_dataset(data, n_sources=40), tmp_path / "leaf-map.json")
    expected = tmp_path / "expected.json"
    # a first rebuild records the hashes (what the laptop does once and commits)
    first = subprocess.run([sys.executable, "scripts/11_make_grouped_splits.py", "--root", str(data), "--leaf-map", str(lm), "--out", str(tmp_path / "s"),
                            "--results", str(tmp_path / "r.json"), "--expected", str(expected), "--write-expected"], cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr

    r = subprocess.run(_cmd(data, work, lm, expected), cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    st = {s["name"]: s["state"] for s in json.loads((work / "results" / "run_grouped_status.json").read_text())["stages"]}
    assert set(st.values()) == {"done"} and len(st) == 5, st
    for arch in ("mobilenet_v2", "efficientnet_b3"):
        for split in ("leaf", "photo"):
            run = json.loads((work / "results" / "runs" / f"{arch}__{split}__s0__base.json").read_text())
            m = run["metrics"]
            assert m["test"]["evaluated_once_after_final_epoch"] and m["test"]["leaf_mapped"]["n"] >= 0
            assert run["config"]["split_meta"]["split_hash"]                       # the run is tied to the exact split
    z = zipfile.ZipFile(work / "export" / "agriscan_grouped_results.zip").namelist()
    assert "splits/leaf_split.csv.gz" in z and not any(n.endswith((".pt", ".png", ".jpg")) for n in z)
    assert (work / "export" / "models" / "mobilenet_v2__leaf__s0__base.pt").exists()
    r2 = subprocess.run(_cmd(data, work, lm, expected), cwd=ROOT, capture_output=True, text=True)   # second run: all done
    assert r2.returncode == 0 and "done_earlier" in r2.stdout and ">>> " not in r2.stdout

    # a dataset that differs from the one the hashes were recorded for must stop the run before any training
    bad = json.loads(expected.read_text())
    bad["files"] = "0" * 64
    expected.write_text(json.dumps(bad))
    r3 = subprocess.run(_cmd(data, tmp_path / "out_bad", lm, expected), cwd=ROOT, capture_output=True, text=True)
    assert r3.returncode == 1 and "MISMATCH" in r3.stdout
    st3 = {s["name"]: s["state"] for s in json.loads((tmp_path / "out_bad" / "results" / "run_grouped_status.json").read_text())["stages"]}
    assert st3 == {"build_grouped_splits": "FAILED"}
