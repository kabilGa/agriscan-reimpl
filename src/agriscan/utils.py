"""Helpers shared by every experiment, so each results JSON is traceable."""
import datetime as dt
import json
import platform
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


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_info() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
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
