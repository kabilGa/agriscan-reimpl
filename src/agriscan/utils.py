"""Helpers shared by every experiment, so each results JSON is traceable."""
import datetime as dt
import json
import platform
from importlib import metadata
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    # Seeds all RNGs we use. This does not make GPU runs bit-identical
    # (some CUDA kernels are non-deterministic); it makes them comparable.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    """Commit of the CODE (not of the working directory); '+dirty' if uncommitted changes exist.

    On Kaggle there is no .git, so fall back to BUNDLE_INFO.json written by make_kaggle_bundle.py.
    """
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True, timeout=10)  # noqa: E731
        commit = run("rev-parse", "--short", "HEAD").stdout.strip()
        if commit:
            return commit + ("+dirty" if run("status", "--porcelain").stdout.strip() else "")
    except Exception:  # noqa: BLE001
        pass
    info = REPO_ROOT / "BUNDLE_INFO.json"
    if info.exists():
        return json.loads(info.read_text()).get("commit", "unknown")
    return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def env_info() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _pkg_version("torchvision"),
        "numpy": _pkg_version("numpy"),
        "pillow": _pkg_version("pillow"),
        "ultralytics": _pkg_version("ultralytics"),
        "cuda": torch.version.cuda if cuda else None,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "git_commit": _git_commit(),
    }


def save_results(path: str | Path, *, name: str, seed: int, config: dict, metrics: dict) -> Path:
    """Write one run to JSON. Every number we report must come from a file like this."""
    record = {
        "name": name,
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "config": config,
        "env": env_info(),
        "metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path
