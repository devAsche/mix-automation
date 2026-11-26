"""Draft docs/evaluations/issues.yaml entries from listening notes.

Input format (same as docs/evaluations/2026-11-26_曲の評価.txt):

    ・曲名
    全体的に…                 -> scope: global
    0:20～「歌詞」の「し」が…  -> time_sec: 20, lyric: 歌詞
    2:24、2:26の…             -> one issue per time

    python scripts/evaluations/parse_notes.py notes.txt > draft.yaml
    python scripts/evaluations/parse_notes.py notes.txt -o docs/evaluations/draft.yaml

category / direction are keyword guesses (marked `draft: true`); a human must review them before
merging into issues.yaml. `quote` is always the user's line verbatim.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):([0-5]\d)(?:\.(\d+))?")
LYRIC_RE = re.compile(r"「([^「」]+)」")

# (regex, category, direction) — first match wins; order matters (specific before generic).
RULES: list[tuple[str, str, str]] = [
    (r"歯擦音|サ行", "sibilance", "too_strong"),
    (r"ディレイ.*拍|拍に合って", "delay_offbeat", "off_grid"),
    (r"処理抜け|抜けてる", "fx_dropout", "missing_fx"),
    (r"ハモリ.*(主張|目立|強)", "harmony_prominent", "too_strong"),
    (r"アタック", "consonant_attack", "too_strong"),
    (r"一段.*奥|奥に行く", "step_change", "too_far"),
    (r"載って", "sits_on_top", "not_blended"),
    (r"リバーブ|ディレイ|空間", "space_excess", "too_much"),
    (r"エアリー|ジリ", "airy_fizzy", "too_airy"),
    (r"こもり|こもって|こもる", "muddy", "too_muddy"),
    (r"埋も|聞こえ(ない|づら|にく)|負けて", "buried", "buried"),
    (r"軽い|薄い|細い", "thin_light", "too_light"),
    (r"距離", "distance", "too_far"),
    (r"つぶれ|潰れ", "sibilance", "crushed"),  # consonant crushed; sibilant is the common case, review
]
DIRECTION_OVERRIDES: list[tuple[str, str]] = [
    (r"つぶれ|潰れ", "crushed"),
    (r"ジリ", "fizzy"),
    (r"過剰", "too_much"),
]
SKIP_LINES = {"全体的に", "全体的には"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("アンド", "&").replace("and", "&")
    return re.sub(r"[\s・_\-]", "", s)


def known_song_names() -> dict[str, str]:
    """normalized -> canonical song_name, from data/raw folder names."""
    names: dict[str, str] = {}
    raw = ROOT / "data" / "raw"
    if raw.is_dir():
        for d in raw.iterdir():
            if d.is_dir():
                names[_norm(d.name)] = d.name
    return names


def canonical_song(heading: str, known: dict[str, str]) -> str:
    return known.get(_norm(heading), heading.strip())


def classify(text: str) -> tuple[str | None, str | None]:
    for pat, cat, direction in RULES:
        if re.search(pat, text):
            for opat, odir in DIRECTION_OVERRIDES:
                if re.search(opat, text):
                    direction = odir
                    break
            return cat, direction
    return None, None


def _prefix(song: str, used: set[str]) -> str:
    ascii_part = re.sub(r"[^a-z]", "", unicodedata.normalize("NFKC", song).lower())[:3]
    base = ascii_part or f"s{len(used) + 1}"
    p = base
    k = 2
    while p in used:
        p = f"{base}{k}"
        k += 1
    used.add(p)
    return p


def parse_notes(text: str, known: dict[str, str] | None = None) -> list[dict]:
    known = known_song_names() if known is None else known
    issues: list[dict] = []
    song: str | None = None
    prefix = ""
    used: set[str] = set()
    n_global = n_timed = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("・", "･")):
            song = canonical_song(line.lstrip("・･ "), known)
            prefix = _prefix(song, used)
            n_global = n_timed = 0
            continue
        if song is None or line in SKIP_LINES:
            continue
        cat, direction = classify(line)
        lyric_m = LYRIC_RE.search(line)
        times = [int(m.group(1)) * 60 + int(m.group(2)) + (float("0." + m.group(3)) if m.group(3) else 0.0)
                 for m in TIME_RE.finditer(line)]
        base = {"song": song}
        tail = {"category": cat, "direction": direction, "draft": True, "quote": line}
        if not times:
            n_global += 1
            issues.append({"id": f"{prefix}-g{n_global}", **base, "scope": "global", **tail})
            continue
        approx = bool(re.search(r"くらい|あたり|頃|直前|直後", line))
        for t in times:
            n_timed += 1
            item = {"id": f"{prefix}-{n_timed}", **base, "time_sec": round(t, 2)}
            if approx:
                item["approx"] = True
            if lyric_m:
                item["lyric"] = lyric_m.group(1)
            issues.append({**item, **tail})
    return issues


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notes", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="write YAML here instead of stdout")
    args = ap.parse_args()

    issues = parse_notes(args.notes.read_text(encoding="utf-8-sig"))
    doc = {"schema_version": 1, "source": args.notes.as_posix(), "issues": issues}
    out = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200)
    if args.out:
        args.out.write_text(out, encoding="utf-8")
        print(f"{len(issues)} issues -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
