"""README claims that depend on measured numbers must agree with the JSON files, and its generated blocks must be current."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
pytestmark = pytest.mark.skipif(not (ROOT / "results" / "step8_grouped_splits.json").exists(), reason="no committed results in this checkout")


def test_uncovered_share_in_the_prose_matches_the_json():
    cov = json.loads((ROOT / "results" / "step8_grouped_splits.json").read_text())["metrics"]["leaf_coverage"]["fraction"]
    assert f"About {round(100 * (1 - cov))}% of the files are not covered by the leaf map" in README


def test_class_coverage_in_the_prose_matches_the_json():
    zero = json.loads((ROOT / "results" / "step8_grouped_splits.json").read_text())["metrics"]["leaf_coverage"]["classes_with_zero_coverage"]
    n = json.loads((ROOT / "results" / "step1_inspect.json").read_text())["metrics"]["n_classes"]
    assert f"**{len(zero)} of the {n} classes have no leaf id at all**" in README
    assert f"the headline covers only the other {n - len(zero)} classes" in README


def test_generated_blocks_are_current(tmp_path):
    for script, marker in (("10_results_table.py", "RESULTS"), ("13_grouped_table.py", "GROUPED")):
        out = tmp_path / f"{marker}.md"
        r = subprocess.run([sys.executable, f"scripts/{script}", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        block = README.split(f"<!-- {marker}:START -->")[1].split(f"<!-- {marker}:END -->")[0]
        assert out.read_text(encoding="utf-8").strip() in block, f"README {marker} block is stale: rerun scripts/{script} --readme README.md"


def test_headline_rule_and_doubts_are_stated():
    for needle in ("headline number is the accuracy on the leaf-mapped subset", "One whole class has no UUID", "` copy` files carry their own UUID",
                   "does **not** remove same-leaf or same-plant leakage", "lab-condition numbers", "never committed"):
        assert needle in README, needle
