"""Shared helpers for scripts/analysis (render folder lookup, sections, audio, time parsing).

Every script in this folder prints short tables only; keep it that way so results can be
pasted into reports without trimming.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "data" / "outputs"
SR = 44100


def resolve_render_dir(arg: str) -> Path:
    """Accept a render folder path, or a song name -> newest `data/outputs/<song>_*` with a report."""
    p = Path(arg)
    if p.is_dir() and (p / "mix_report.json").exists():
        return p
    cands = [d for d in OUTPUTS.glob(f"{arg}*") if (d / "mix_report.json").exists()]
    if not cands:
        raise SystemExit(f"render folder not found for '{arg}' (looked in {OUTPUTS})")
    return max(cands, key=lambda d: (d / "mix_report.json").stat().st_mtime)


def load_report(render_dir: Path) -> dict:
    return json.loads((render_dir / "mix_report.json").read_text(encoding="utf-8"))


def sections(report: dict) -> list[dict]:
    return list(report.get("sections") or [])


def parse_time(text: str) -> float:
    """'0:20' / '2:34.5' / '83' / '1:02:03' -> seconds. Accepts full-width colon and digits."""
    t = text.strip().translate(str.maketrans("０１２３４５６７８９：．", "0123456789:."))
    t = t.rstrip("～~〜-")
    parts = t.split(":")
    if not all(re.fullmatch(r"\d+(\.\d+)?", x) for x in parts):
        raise ValueError(f"bad time: {text!r}")
    sec = 0.0
    for x in parts:
        sec = sec * 60.0 + float(x)
    return sec


def fmt_time(sec: float) -> str:
    m, s = divmod(max(0.0, sec), 60.0)
    return f"{int(m)}:{s:04.1f}"


def section_at(report: dict, t: float) -> dict | None:
    for s in sections(report):
        if s["start_sec"] <= t < s["end_sec"]:
            return s
    return None


def load_ch_first(path: Path, sr: int = SR) -> np.ndarray:
    """Load audio as float32 (channels, samples)."""
    from mix_automation.io.audio_loader import load_audio

    audio, _ = load_audio(Path(path), target_sr=sr, mono=False)
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.shape[0] > a.shape[1]:
        a = a.T
    return a


def sum_to_length(arrays: list[np.ndarray]) -> np.ndarray:
    n = max(a.shape[1] for a in arrays)
    ch = max(a.shape[0] for a in arrays)
    out = np.zeros((ch, n), dtype=np.float32)
    for a in arrays:
        aa = a if a.shape[0] == ch else np.repeat(a[:1], ch, axis=0)
        out[:, : aa.shape[1]] += aa
    return out


def raw_vocal_stems(report: dict) -> list[Path]:
    """Input vocal stem files recorded in the report (searched under the song folder)."""
    ia = report.get("input_assets") or {}
    folder = Path(ia.get("song_folder", ""))
    found: list[Path] = []
    for st in ia.get("vocal_stems") or []:
        hits = list(folder.rglob(st["file"])) if folder.is_dir() else []
        if hits:
            found.append(hits[0])
    return found


def first_glob(render_dir: Path, pattern: str) -> Path | None:
    return next(iter(sorted(render_dir.glob(pattern))), None)


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(_disp_len(row[i]) for row in cells) for i in range(len(headers))]
    for k, row in enumerate(cells):
        print("  ".join(_pad(c, widths[i]) for i, c in enumerate(row)))
        if k == 0:
            print("  ".join("-" * w for w in widths))


def _disp_len(s: str) -> int:
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)


def _pad(s: str, w: int) -> str:
    return s + " " * (w - _disp_len(s))
