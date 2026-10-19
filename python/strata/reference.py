"""Pinned reference checkout guard shared by every entrypoint."""

from pathlib import Path
import subprocess
import sys


REFERENCE_SHA = "b4451682abac651f26046f8d27467de3f007e1e7"


def load_reference(path):
    path = Path(path).resolve()
    git = ["git", "-c", f"safe.directory={path}", "-C", str(path)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    if revision != REFERENCE_SHA or dirty:
        raise ValueError(f"A clean reference checkout at {REFERENCE_SHA} is required")
    sys.path.insert(0, str(path))
