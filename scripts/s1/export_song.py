"""R5 B2: Studio One .song from a v8 engine output folder (Vocal CorrEQ track + sends + Inst + muted Ref).

    .venv\\Scripts\\python.exe scripts\\s1\\export_song.py <v8 out dir> [--out <dir>] [--correction-eq <json>] [--no-verify]

Writes <out>/<slug>_v8r5.song, <slug>_v8r5_orig.song, s1_media/*.wav (float) and export_report.json.
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
from mix_automation.v8.s1_export import export_v8_song  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("v8_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--correction-eq", type=Path, default=None, help="correction EQ JSON for the Vocal CorrEQ insert")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = ceq.load(a.correction_eq) if a.correction_eq else None
    rep = export_v8_song(a.v8_dir, a.out, cfg, verify=not a.no_verify)
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
