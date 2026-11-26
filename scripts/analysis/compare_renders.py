"""Compare two renders of the same song (reproduction check / A-B sanity): LUFS, null residual, report numbers.

    python scripts/analysis/compare_renders.py data/archive/gemini_v7_20261126/エバ_auto C:/MIX_automation_gemini/outputs/エバ_auto

Prints integrated LUFS / true-peak-ish max for Master and VocalBus, the null-test residual
(difference energy relative to A, dB; -inf = bit identical), and every numeric field of
mix_report.json that differs by more than --tol (only the differing ones, plus a count).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SR, first_glob, load_ch_first, load_report, print_table, resolve_render_dir  # noqa: E402

SKIP_KEYS = {"git_commit", "git_dirty", "profiles_json", "profiles_sha256", "engine"}


def _flatten(obj: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float)) and math.isfinite(obj):
        out[prefix] = float(obj)
    return out


def _lufs(a: np.ndarray) -> float:
    import pyloudnorm as pyln

    return float(pyln.Meter(SR).integrated_loudness(a.T.astype(np.float64)))


def _audio_row(name: str, pa: Path | None, pb: Path | None) -> list[object]:
    if pa is None or pb is None:
        return [name, "missing", "", "", "", ""]
    a, b = load_ch_first(pa), load_ch_first(pb)
    n = min(a.shape[1], b.shape[1])
    ch = min(a.shape[0], b.shape[0])
    aa, bb = a[:ch, :n], b[:ch, :n]
    ea = float(np.sum(aa.astype(np.float64) ** 2))
    ed = float(np.sum((aa.astype(np.float64) - bb) ** 2))
    resid = 10 * math.log10(ed / ea) if ed > 0 and ea > 0 else float("-inf")
    return [
        name,
        f"{_lufs(aa):.2f}",
        f"{_lufs(bb):.2f}",
        f"{float(np.max(np.abs(aa))):.4f}/{float(np.max(np.abs(bb))):.4f}",
        f"{resid:.1f}",
        f"{a.shape[1] - b.shape[1]:+d}",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", help="render A (folder or song name)")
    ap.add_argument("b", help="render B (folder or song name)")
    ap.add_argument("--tol", type=float, default=1e-3, help="absolute tolerance for report numbers")
    ap.add_argument("--max-rows", type=int, default=25)
    args = ap.parse_args()

    da, db = resolve_render_dir(args.a), resolve_render_dir(args.b)
    ra, rb = load_report(da), load_report(db)
    print(f"A: {da}\nB: {db}")
    rows = [
        _audio_row("Master", first_glob(da, "*_AutoMIX_Master.wav"), first_glob(db, "*_AutoMIX_Master.wav")),
        _audio_row("VocalBus", first_glob(da, "*_AutoMIX_VocalBus.wav"), first_glob(db, "*_AutoMIX_VocalBus.wav")),
    ]
    print_table(["file", "LUFS A", "LUFS B", "peak A/B", "null dB", "len diff"], rows)

    fa, fb = _flatten(ra), _flatten(rb)
    common = sorted(set(fa) & set(fb))
    diffs = [(k, fa[k], fb[k]) for k in common if abs(fa[k] - fb[k]) > args.tol]
    diffs.sort(key=lambda x: -abs(x[1] - x[2]))
    only = len(set(fa) ^ set(fb))
    print(f"\nreport numbers: {len(common)} compared, {len(diffs)} differ (> {args.tol}), {only} present in only one")
    if diffs:
        print_table(["field", "A", "B", "B-A"],
                    [[k, f"{x:.4g}", f"{y:.4g}", f"{y - x:+.4g}"] for k, x, y in diffs[: args.max_rows]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
