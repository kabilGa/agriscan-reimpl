"""scripts/14 on a synthetic dataset: photo split overlaps by leaf, leaf split does not, and both computations agree."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from synth import make_grouped_dataset, write_leaf_map

ROOT = Path(__file__).resolve().parents[1]


def _run(script, *args):
    r = subprocess.run([sys.executable, f"scripts/{script}", *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def test_overlap_check_agrees_with_the_split_build(tmp_path):
    data = tmp_path / "data"
    lm = write_leaf_map(make_grouped_dataset(data, n_sources=60), tmp_path / "leaf-map.json")
    splits, step8, exp = tmp_path / "splits", tmp_path / "step8.json", tmp_path / "exp.json"
    _run("11_make_grouped_splits.py", "--root", data, "--leaf-map", lm, "--out", splits, "--results", step8, "--expected", exp, "--write-expected")
    res = tmp_path / "overlap.json"
    _run("14_leaf_overlap_check.py", "--splits-dir", splits, "--leaf-map", lm, "--expected", exp, "--stored", step8, "--results", res)
    m = json.loads(res.read_text())["metrics"]
    assert m["photo"]["test"]["share_with_same_leaf_in_train"] > 0.3          # photo grouping leaves same-leaf images in training
    assert m["leaf"]["test"]["share_with_same_leaf_in_train"] == 0.0 and m["leaf"]["test"]["n_with_same_leaf_in_train"] == 0
    assert all(m[s][p]["agrees_with_split_build"] for s in ("photo", "leaf") for p in ("val", "test"))
    assert m["photo"]["test"]["same_leaf_train_images_per_image"]["mean"] > 0
    # a leaf map that differs from the one the hashes were recorded for is refused
    bad = json.loads(exp.read_text())
    bad["leaf_map_sha256"] = "0" * 64
    exp.write_text(json.dumps(bad))
    r = subprocess.run([sys.executable, "scripts/14_leaf_overlap_check.py", "--splits-dir", str(splits), "--leaf-map", str(lm), "--expected", str(exp),
                        "--results", str(tmp_path / "x.json")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout)
