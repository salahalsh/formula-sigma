"""Run-manifest writer.

Every experiment script must call write_manifest() at the end so that
e04_reproducibility can hash + verify outputs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "n/a"


def write_manifest(
    *,
    experiment: str,
    out_dir: Path,
    seed: int,
    inputs: Iterable[Path | str],
    outputs: Iterable[Path | str],
    runtime_sec: float,
    extra: Mapping | None = None,
) -> Path:
    """Write `run_manifest.json` capturing reproducibility metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": experiment,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "runtime_sec": round(runtime_sec, 3),
        "git_sha": _git_sha(Path(__file__).resolve().parents[2]),
        "inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in inputs],
        "outputs": [{"path": str(p), "sha256": sha256_file(p)} for p in outputs if Path(p).exists()],
        "key_libs": _key_lib_versions(),
        "extra": dict(extra) if extra else {},
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _key_lib_versions() -> dict[str, str]:
    libs = ["numpy", "pandas", "scipy", "statsmodels", "sklearn", "pyDOE3",
            "skopt", "pymoo", "matplotlib", "seaborn"]
    out = {}
    for name in libs:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except ImportError:
            out[name] = "not installed"
    return out
