"""The committed notebooks are exactly what scripts/make_notebooks.py generates, and are valid notebooks."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_generated_and_valid():
    before = {n: (ROOT / "notebooks" / n).read_text(encoding="utf-8").replace("\r\n", "\n") for n in ("kaggle_run_all.ipynb", "kaggle_run_grouped.ipynb")}
    r = subprocess.run([sys.executable, "scripts/make_notebooks.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for n, old in before.items():
        new = (ROOT / "notebooks" / n).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert new == old, f"{n} is out of date: run scripts/make_notebooks.py"
        nb = json.loads(new)
        assert nb["nbformat"] == 4 and all("outputs" in c for c in nb["cells"] if c["cell_type"] == "code")
    grouped = json.loads(before["kaggle_run_grouped.ipynb"])
    src = "\n".join("".join(c["source"]) for c in grouped["cells"])
    assert "run_grouped.py" in src and "run_all.py" not in src and "for depth in (1, 2, 3)" in src
