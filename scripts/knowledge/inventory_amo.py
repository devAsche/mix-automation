"""Inventory of utattemita_practicesite_reference/Amo_* (R1 step 0).

Prints one row per song: cover/original length, practice sets (pre/post correction)
with track counts per role, inst presence, and the song titles that appear in file names.

    .venv\\Scripts\\python.exe scripts\\knowledge\\inventory_amo.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "utattemita_practicesite_reference"

PRE = ("補正前", "補正無", "補正なし", "補正処理なし", "修正前", "修正無", "修正なし", "素音源")
POST = ("補正後", "補正語", "補正有", "補正あり", "補正済", "修正後", "修正有", "修正あり")

ROLES = [  # order matters: first match wins
    ("inst", r"inst|オフボ|カラオケ"),
    ("dbl", r"ダブ|dbl|doubl"),
    ("oct", r"オク|oct|okus"),
    ("hamo", r"ハモ|はも|hamo|hami|harm|^h\d"),
    ("speech", r"セリフ"),
    ("sub", r"サブ"),
    ("main", r"メイン|main|^m\d"),
]


def duration(p: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return float(out)
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def role(name: str) -> str:
    n = name.lower()
    # "サブ上ハモ" etc. count as hamo; plain "サブ1" as sub
    for r, pat in ROLES:
        if re.search(pat, n):
            return r
    return "other"


def set_kind(rel: Path) -> str:
    s = str(rel)
    if any(k in s for k in PRE):
        return "pre"
    if any(k in s for k in POST):
        return "post"
    if "MIDI" in s:
        return "midi"
    return "flat"


def fmt(sec: float | None) -> str:
    if sec is None:
        return "-"
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    for song in sorted(ROOT.glob("Amo_*")):
        cover = next(song.glob("cover_*"), None)
        orig = next(song.glob("original_*"), None)
        sets: dict[str, dict[str, int]] = {}
        lens: dict[str, list[float]] = {}
        titles: set[str] = set()
        inst = []
        for f in sorted((song / "practice").rglob("*.wav")):
            rel = f.relative_to(song / "practice")
            k = set_kind(rel)
            r = role(f.stem)
            if r == "inst":
                inst.append(f)
            sets.setdefault(k, {}).setdefault(r, 0)
            sets[k][r] += 1
            d = duration(f)
            if d:
                lens.setdefault(k, []).append(d)
            m = re.match(r"(.+?)\s*[-_]", f.stem)
            if m and not re.search(r"補正|修正|^Amo$|48kHz|BabyFace|queen", m.group(1), re.I):
                titles.add(m.group(1).strip())
        row = {
            "song": song.name.removeprefix("Amo_"),
            "cover": fmt(duration(cover)) if cover else "無",
            "original": fmt(duration(orig)) if orig else "無",
            "practice": {k: dict(sorted(v.items())) for k, v in sets.items()},
            "practice_len": {k: fmt(max(v)) for k, v in lens.items()},
            "inst": fmt(duration(inst[0])) if inst else "無",
            "titles_in_files": sorted(titles),
        }
        rows.append(row)
        ps = " / ".join(
            f"{k}:" + ",".join(f"{r}{n}" for r, n in v.items()) + f"({row['practice_len'].get(k, '-')})"
            for k, v in row["practice"].items()
        )
        print(f"{row['song']:<16} cover {row['cover']:>5} orig {row['original']:>5} inst {row['inst']:>4} | {ps} | {','.join(row['titles_in_files'])}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
