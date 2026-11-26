"""Per-section vocal/backing balance: V/I loudness ratios, Side/Mid widths, partial-loudness margin.

    python scripts/analysis/section_balance.py エバ
    python scripts/analysis/section_balance.py エバ --at 0:20 2:34 --win 4
    python scripts/analysis/section_balance.py DIR --vocal v.wav --inst i.wav    # e.g. pro reference stems

Defaults: vocal = *_AutoMIX_VocalBus.wav, inst = Media/01_offvocal_unmasked.wav.
Metrics come from mix_automation.analysis.relative_profile.compute_section_relative_profile,
so they match the anchor/full-song numbers stored in mix_report.json:
  kVI = K-weighted V/I (dB), midVI = same on the Mid channel, bbVI = broadband RMS V/I,
  vSM / iSM = vocal / inst Side-minus-Mid (dB), PLm = partial loudness margin (dB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    SR,
    first_glob,
    fmt_time,
    load_ch_first,
    load_report,
    parse_time,
    print_table,
    resolve_render_dir,
    sections,
)


def _stereo(a):
    import numpy as np

    return a if a.shape[0] == 2 else np.repeat(a[:1], 2, axis=0)


def main() -> int:
    from mix_automation.analysis.relative_profile import compute_section_relative_profile

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("render", help="render folder or song name")
    ap.add_argument("--vocal", type=Path, default=None)
    ap.add_argument("--inst", type=Path, default=None)
    ap.add_argument("--at", nargs="*", default=[], help="windows starting at these times (m:ss～) instead of sections")
    ap.add_argument("--win", type=float, default=4.0, help="window length in seconds for --at")
    args = ap.parse_args()

    render_dir = resolve_render_dir(args.render)
    report = load_report(render_dir)
    v_path = args.vocal or first_glob(render_dir, "*_AutoMIX_VocalBus.wav")
    i_path = args.inst or (render_dir / "Media" / "01_offvocal_unmasked.wav")
    if v_path is None or not Path(i_path).exists():
        raise SystemExit("vocal/inst file not found; pass --vocal/--inst")
    voc = _stereo(load_ch_first(v_path))
    inst = _stereo(load_ch_first(i_path))
    n = min(voc.shape[1], inst.shape[1])

    if args.at:
        spans = [(fmt_time(c), c, c + args.win) for c in map(parse_time, args.at)]
    else:
        spans = [(f"{s['index']:02d} {s['label']}", s["start_sec"], s["end_sec"]) for s in sections(report)]

    rows = []
    for name, t0, t1 in spans:
        a, b = max(0, int(t0 * SR)), min(n, int(t1 * SR))
        if b - a < SR // 2:
            continue
        try:
            p = compute_section_relative_profile(voc[:, a:b], inst[:, a:b], sr=SR)
        except Exception as exc:  # noqa: BLE001 - one silent section must not kill the table
            rows.append([name, f"{fmt_time(t0)}-{fmt_time(t1)}", f"error: {type(exc).__name__}"] + [""] * 5)
            continue
        rows.append([
            name,
            f"{fmt_time(t0)}-{fmt_time(t1)}",
            f"{p.k_lufs_vi_ratio_db:+.1f}",
            f"{p.mid_k_lufs_vi_ratio_db:+.1f}",
            f"{p.broadband_vi_ratio_db:+.1f}",
            f"{p.vocal_side_mid_db:+.1f}",
            f"{p.inst_side_mid_db:+.1f}",
            f"{p.partial_loudness_margin_db:+.1f}",
        ])

    print(f"render: {render_dir.name}  vocal: {Path(v_path).name}  inst: {Path(i_path).name}")
    print_table(["span", "range", "kVI", "midVI", "bbVI", "vSM", "iSM", "PLm"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
