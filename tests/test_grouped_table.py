"""scripts/13: the grouped-protocol tables on (a) a smoke run's outputs and (b) an empty results folder."""
import json
import re
import subprocess
import sys
from pathlib import Path

from synth import make_grouped_dataset, write_leaf_map

ROOT = Path(__file__).resolve().parents[1]


def _table(results, out, readme=None):
    cmd = [sys.executable, "scripts/13_grouped_table.py", "--results", str(results), "--out", str(out)] + (["--readme", str(readme)] if readme else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    return out.read_text(encoding="utf-8")


def test_empty_results_say_not_run_yet(tmp_path):
    (tmp_path / "res" / "runs").mkdir(parents=True)
    text = _table(tmp_path / "res", tmp_path / "t.md")
    assert "not run yet" in text and "no headline number" in text


def test_tables_from_a_smoke_run_match_the_json(tmp_path):
    data, work = tmp_path / "data", tmp_path / "out"
    lm = write_leaf_map(make_grouped_dataset(data, n_sources=40), tmp_path / "leaf-map.json")
    r = subprocess.run([sys.executable, "scripts/run_grouped.py", "--preset", "smoke", "--data", str(data), "--workdir", str(work), "--leaf-map", str(lm),
                        "--expected", str(tmp_path / "none.json")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    readme = tmp_path / "README.md"
    readme.write_text("top\n<!-- GROUPED:START -->\nSTALE-CONTENT\n<!-- GROUPED:END -->\nbottom\n", encoding="utf-8")
    text = _table(work / "results", tmp_path / "t.md", readme)
    leaf = json.loads((work / "results" / "runs" / "mobilenet_v2__leaf__s0__base.json").read_text())["metrics"]["test"]
    row = next(line for line in text.splitlines() if line.startswith("| mobilenet_v2 | leaf+photo-grouped (primary)"))
    assert f"{leaf['acc']:.4f}" in row and f"({leaf['leaf_mapped']['n']})" in row
    assert "**Headline" in text and "Leaf-map coverage" in text
    new = readme.read_text(encoding="utf-8")
    assert new.startswith("top\n<!-- GROUPED:START -->") and new.rstrip().endswith("bottom") and "STALE-CONTENT" not in new
    assert not re.search(r"\b(nan|inf)\b", text.lower())
