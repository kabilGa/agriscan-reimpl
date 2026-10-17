"""The documentation must not contain broken paths, stray control characters, a byte-order mark, or the CV claims we decided not to make."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "DECISIONS.md", "LICENSES.md", "KAGGLE_RUN.md"]
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TRUNCATED = re.compile(r"(?<![\w/.\-])(esults|cripts|plits|rtifacts|otebooks|ests)/")      # a directory name that lost its first letter
PATH = re.compile(r"`((?:results|scripts|splits|src|tests|notebooks)/[^`\s]+)`")
SKIP = ("*", "<", ">", "{", "...")                                                       # patterns and placeholders, not real paths
BOM = b"\xef\xbb\xbf"
FIRST_LINE = "Re-implementation of a 2026 team project; original files lost; this repo covers my vision and web part."


def text_problems(text: str) -> list:
    """Everything wrong with one document's text (LF-normalised): control characters, lone CR, truncated paths, odd backticks."""
    out = []
    if CTRL.search(text):
        out.append("control character")
    if "\r" in text:
        out.append("lone carriage return (a PowerShell backtick-r escape?)")
    in_fence = False
    for i, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if TRUNCATED.search(line):
            out.append(f"line {i}: broken path fragment {TRUNCATED.search(line).group(0)!r}")
        if line.count("`") % 2:
            out.append(f"line {i}: odd number of backticks")
    return out


def _text(name):
    return (ROOT / name).read_bytes().decode("utf-8").replace("\r\n", "\n")                # CRLF on Windows checkouts is fine


def _repo_text_files():
    skip = {".venv", ".git", "artifacts", "export", "agriscan_results", "__pycache__", ".pytest_cache"}
    private = {"LEARNING_NOTES.md", "INTERVIEW_QA.md"}                                    # local study notes: git-ignored, never published
    for p in ROOT.rglob("*"):
        if p.is_file() and p.name not in private and not (skip & set(p.relative_to(ROOT).parts)) and (p.suffix in {".md", ".py", ".json", ".toml", ".txt", ".ipynb"} or p.name in {".gitignore", ".gitattributes"}):
            yield p


def test_detector_catches_the_original_damage_and_accepts_the_fix():
    # What PowerShell produced when "`r" in a double-quoted string was read as a carriage return, and the closing backtick vanished:
    damaged = "(" + "\r" + "esults/extra_leaf_overlap_check.json recomputes this independently)"
    fixed = "(`results/extra_leaf_overlap_check.json` recomputes this independently)"
    assert text_problems(damaged), "the damaged text must be reported"
    assert text_problems(damaged.replace("\r", "\n")), "still reported if the CR was normalised to a newline"
    assert not text_problems(fixed)
    assert text_problems("a `broken") and text_problems("text with a \x07 bell") and text_problems("see `scripts/x.py` and cripts/y.py")


def test_no_byte_order_mark_anywhere():
    bad = [str(p.relative_to(ROOT)) for p in _repo_text_files() if p.read_bytes().startswith(BOM)]
    assert not bad, f"UTF-8 BOM (PowerShell Set-Content -Encoding UTF8 adds one): {bad}"


def test_readme_first_line_is_exactly_the_statement():
    assert _text("README.md").split("\n")[0] == FIRST_LINE


@pytest.mark.parametrize("name", DOCS)
def test_documents_have_no_text_problems(name):
    assert not text_problems(_text(name)), f"{name}: {text_problems(_text(name))}"


@pytest.mark.parametrize("name", DOCS)
def test_backticked_repo_paths_exist(name):
    missing = [p for p in PATH.findall(_text(name)) if not any(s in p for s in SKIP) and not (ROOT / p.rstrip(".,;:)")).exists()]
    assert not missing, f"{name}: paths that do not exist: {missing}"


def test_cv_wording_and_the_numbers_it_quotes():
    readme = _text("README.md")
    assert "96.7% validation accuracy the original team project reported" in readme
    assert "96.7% on my CV" not in readme and "96.7% on the CV" not in _text("DECISIONS.md")
    assert "96.7% validation accuracy the original team project reported" in _text("DECISIONS.md")
    sentence = next(line for line in readme.split("\n") if line.startswith("- The AgriScan numbers on my CV"))
    assert "come from this re-implementation's grouped protocol, not from the original April project" in sentence
    assert "31 of 38 classes" in sentence and "single seed, no seed spread measured" in sentence
    accs = []
    for arch in ("mobilenet_v2", "efficientnet_b3"):
        t = json.loads((ROOT / "results" / "runs" / f"{arch}__leaf__s0__base.json").read_text())["metrics"]["test"]["leaf_mapped"]["acc"]
        accs.append(f"{100 * t:.1f}%")
    assert all(a in sentence for a in accs), (accs, sentence)
