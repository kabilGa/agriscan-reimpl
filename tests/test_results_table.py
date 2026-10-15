"""The results tables must match the JSON files (recomputed independently here). Skipped if results/ is absent."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
pytestmark = pytest.mark.skipif(not (R / "step1_leakage.json").exists(), reason="no committed results in this checkout")


def _run(tmp_path, readme=None):
    cmd = [sys.executable, "scripts/10_results_table.py", "--out", str(tmp_path / "R.md")]
    if readme:
        cmd += ["--readme", str(readme)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return (tmp_path / "R.md").read_text(encoding="utf-8")


def _acc(arch, split, tag="base"):
    out = []
    for p in (R / "runs").glob("*.json"):
        c = json.loads(p.read_text())["config"]
        if (c["arch"], c["split"], c["tag"]) == (arch, split, tag):
            out.append(json.loads(p.read_text())["metrics"]["val_acc"])
    return out


def test_tables_match_the_json_files(tmp_path):
    text = _run(tmp_path)
    assert not re.search(r"\b(nan|inf)\b", text.lower())
    for arch in ("mobilenet_v2", "efficientnet_b3"):
        for split in ("clean", "original"):
            a = _acc(arch, split)
            row = next(line for line in text.splitlines() if line.startswith(f"| {arch} |") and f"| {split} |" in line)
            assert f"{np.mean(a):.4f}" in row and f"| {len(a)} (seeds" in row
            if len(a) > 1:
                assert f"{min(a):.4f}-{max(a):.4f}" in row
    leak = json.loads((R / "step1_leakage.json").read_text())["metrics"]["centered"]["valid_to_train"]["frac_at_least"]["0.9"]
    row = next(line for line in text.splitlines() if line.startswith("| 0.9 |"))
    assert f"{100 * leak:.1f}%" in row
    for tag in ("frozen_only", "no_aug", "size128"):
        assert f"{_acc('mobilenet_v2', 'clean', tag)[0]:.4f}" in next(line for line in text.splitlines() if line.startswith(f"| {tag} |"))


def test_readme_block_is_replaced_in_place(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("top\n<!-- RESULTS:START -->\nSTALE-CONTENT\n<!-- RESULTS:END -->\nbottom\n", encoding="utf-8")
    text = _run(tmp_path, readme)
    new = readme.read_text(encoding="utf-8")
    assert new.startswith("top\n<!-- RESULTS:START -->") and new.rstrip().endswith("bottom") and "STALE-CONTENT" not in new
    assert "### Classifiers: original split vs clean split" in new and text.strip() in new
