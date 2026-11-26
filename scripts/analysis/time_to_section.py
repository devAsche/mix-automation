"""Map given times (or the issues in docs/evaluations/issues.yaml) to the render's sections.

    python scripts/analysis/time_to_section.py エバ 0:20 1:05
    python scripts/analysis/time_to_section.py data/outputs/エバ_auto --issues docs/evaluations/issues.yaml

Prints one row per time: section index/label/range, position inside the section, the
section's clip gain, and how many Stage-3 FX events overlap that time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    fmt_time,
    load_report,
    parse_time,
    print_table,
    resolve_render_dir,
    section_at,
)


def _fx_overlaps(report: dict, t: float) -> list[str]:
    hits = []
    for ev in report.get("stage3_creative_fx_events") or []:
        s = ev.get("start_sec", ev.get("time_sec"))
        e = ev.get("end_sec", s)
        if s is not None and float(s) - 0.05 <= t <= float(e) + 0.05:
            hits.append(str(ev.get("fx_type", ev.get("kind", ev.get("type", "fx")))))
    return hits


def _issue_times(path: Path, song: str) -> list[tuple[str, float, str]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for it in data.get("issues", []):
        if it.get("song") != song or it.get("scope") == "global":
            continue
        t = it.get("time_sec", it.get("start"))
        if t is not None:
            out.append((str(it.get("id")), float(t), str(it.get("category", ""))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("render", help="render folder or song name (newest data/outputs/<song>_*)")
    ap.add_argument("times", nargs="*", help="times like 0:20 2:34.5")
    ap.add_argument("--issues", type=Path, help="issues.yaml; uses entries whose song matches the report")
    ap.add_argument("--song", help="song name in issues.yaml (default: report song_name)")
    args = ap.parse_args()

    render_dir = resolve_render_dir(args.render)
    report = load_report(render_dir)
    items: list[tuple[str, float, str]] = [("", parse_time(t), "") for t in args.times]
    if args.issues:
        items += _issue_times(args.issues, args.song or report.get("song_name", ""))
    if not items:
        raise SystemExit("no times given (pass times or --issues)")

    print(f"render: {render_dir.name}  engine={report.get('engine', 'v7?')}  sections={len(report.get('sections') or [])}")
    rows = []
    for ident, t, cat in sorted(items, key=lambda x: x[1]):
        s = section_at(report, t)
        fx = _fx_overlaps(report, t)
        if s is None:
            rows.append([ident, fmt_time(t), cat, "-", "(outside)", "-", "-", "-", len(fx)])
            continue
        rows.append([
            ident,
            fmt_time(t),
            cat,
            s["index"],
            s["label"],
            f"{fmt_time(s['start_sec'])}-{fmt_time(s['end_sec'])}",
            f"+{t - s['start_sec']:.1f}s",
            s.get("clip_gain_db"),
            len(fx),
        ])
    print_table(["id", "time", "category", "idx", "section", "range", "pos", "clip_dB", "fx"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
