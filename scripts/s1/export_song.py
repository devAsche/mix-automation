"""R5 B2: Studio One .song from a v8 engine output folder (Vocal CorrEQ track + sends + Inst + muted Ref).

    .venv\\Scripts\\python.exe scripts\\s1\\export_song.py <v8 out dir> [--out <dir>] [--correction-eq <json>] [--tag v8r5] [--no-verify]

Writes <out>/<slug>_<tag>.song, <slug>_<tag>_orig.song, s1_media/*.wav (float) and export_report.json.
Without --correction-eq the Vocal CorrEQ insert gets <v8 out dir>/correction_eq_applied.json (the EQ the
engine applied) when it exists, else flat.
<out> defaults to <v8 out dir>/s1_export; a non-empty folder is never written into (timestamped sibling).
Verification (default on) re-renders the master from the written WAVs and presets with the real plugins
(several minutes for a full song) and re-sets the pre-limiter gain when LUFS differs by > 0.1 LU.
Studio One itself is not started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mix_automation.v8 import correction_eq as ceq  # noqa: E402
from mix_automation.v8.engine import APPLIED_CEQ_NAME  # noqa: E402
from mix_automation.v8.s1_export import EXPORT_TAG, export_v8_song  # noqa: E402


def correction_eq_path(v8_dir: Path, explicit: Path | None) -> Path | None:
    """--correction-eq if given, else the engine's <v8_dir>/correction_eq_applied.json if it exists."""
    if explicit is not None:
        return explicit
    applied = Path(v8_dir) / APPLIED_CEQ_NAME
    return applied if applied.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("v8_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--correction-eq", type=Path, default=None, help="correction EQ JSON for the Vocal CorrEQ insert")
    ap.add_argument("--tag", default=EXPORT_TAG, help="song name suffix: <slug>_<tag>.song (default %(default)s)")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    ceq_path = correction_eq_path(a.v8_dir, a.correction_eq)
    cfg = ceq.load(ceq_path) if ceq_path else None
    print(f"Vocal CorrEQ: {ceq_path or 'flat'}")
    rep = export_v8_song(a.v8_dir, a.out, cfg, verify=not a.no_verify, tag=a.tag)
    print(f"song: {rep['song']}\norig: {rep['song_orig']}\nreport: {Path(rep['out_dir']) / 'export_report.json'}")
    v = rep["verify"]
    if v:
        print("| chain | LUFS diff (LU) | null vs v8 master (dB) |\n|---|---|---|")
        print(f"| plugins + M/S comp | {v['with_ms_compressor']['lufs_diff']:+.3f} | {v['with_ms_compressor']['null_db']:.1f} |")
        print(f"| .song (no M/S comp) | {v['song_chain']['lufs_diff']:+.3f} | {v['song_chain']['null_db']:.1f} |")
        print(f"drive_db (pre-limiter gain): report {v['pre_gain_db_report']} -> used {v['drive_db_used']} "
              f"tries {json.dumps(v['drive_db_tries'])}")
    print("check: " + ("OK" if not rep["check_issues"] else "; ".join(rep["check_issues"])))
    return 0 if not rep["check_issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
