"""Render provenance: which engine, which code and which profiles produced an output.

Recorded into every mix_report.json so v7/v8 A/B comparisons can always be traced back to
an exact commit (and whether the working tree had uncommitted edits at render time).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_commit() -> str | None:
    """Full SHA of HEAD for the checkout this package was imported from (None outside git)."""
    out = _git("rev-parse", "HEAD")
    return out.strip() if out else None


def git_dirty() -> bool | None:
    """True when tracked files have uncommitted changes (untracked files are ignored)."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    if out is None:
        return None
    return bool(out.strip())


def file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_provenance(engine: str, profiles_json_path: Path | str | None) -> dict[str, object]:
    """Fields merged into the top level of mix_report.json."""
    return {
        "engine": engine,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "profiles_json": str(profiles_json_path) if profiles_json_path is not None else None,
        "profiles_sha256": file_sha256(profiles_json_path),
    }
