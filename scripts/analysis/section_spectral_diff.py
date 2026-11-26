"""Per-section (or per-time-window) spectrum change between a vocal before and after processing.

    python scripts/analysis/section_spectral_diff.py エバ                       # raw stems -> VocalBus, by section
    python scripts/analysis/section_spectral_diff.py エバ --pre prepared        # Media/02_mainvocal_prepared.wav -> VocalBus
    python scripts/analysis/section_spectral_diff.py エバ --at 0:20 2:34 --win 1.5
    python scripts/analysis/section_spectral_diff.py DIR --pre a.wav --post b.wav

Columns: gain = overall level change (dB, active frames only); each band = band change minus the
overall change (dB), i.e. pure tonal change. Bands are VOCAL_MIX_BANDS from
mix_automation.analysis.relative_profile so numbers line up with mix_report.json.
Both files are assumed to share the song timeline (true for pipeline outputs and input stems).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    SR,
    first_glob,
    fmt_time,
    load_ch_first,
    load_report,
    parse_time,
    print_table,
    raw_vocal_stems,
    resolve_render_dir,
    sections,
    sum_to_length,
)

N_FFT = 2048
HOP = 1024


def _band_power_frames(mono: np.ndarray, bands: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """(frames, bands) band power and (frames,) total power."""
    n_frames = max(1, 1 + (len(mono) - N_FFT) // HOP)
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in bands.values()]
    bp = np.zeros((n_frames, len(bands)), dtype=np.float64)
    tot = np.zeros(n_frames, dtype=np.float64)
    step = 2048
    for f0 in range(0, n_frames, step):
        idx = np.arange(f0, min(n_frames, f0 + step))
        frames = np.stack([mono[i * HOP : i * HOP + N_FFT] for i in idx])
        if frames.shape[1] < N_FFT:
            frames = np.pad(frames, ((0, 0), (0, N_FFT - frames.shape[1])))
        spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
        tot[idx] = spec.sum(axis=1)
        for b, m in enumerate(masks):
            bp[idx, b] = spec[:, m].sum(axis=1)
    return bp, tot


def _resolve_pre(arg: str, render_dir: Path, report: dict) -> tuple[np.ndarray, str]:
    if arg == "raw":
        stems = raw_vocal_stems(report)
        if not stems:
            raise SystemExit("input vocal stems from mix_report not found; pass --pre <file>")
        return sum_to_length([load_ch_first(p) for p in stems]), f"raw stems x{len(stems)}"
    if arg == "prepared":
        p = render_dir / "Media" / "02_mainvocal_prepared.wav"
        return load_ch_first(p), p.name
    return load_ch_first(Path(arg)), Path(arg).name


def main() -> int:
    from mix_automation.analysis.relative_profile import VOCAL_MIX_BANDS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("render", help="render folder or song name")
    ap.add_argument("--pre", default="raw", help="raw (input stems, default) | prepared | <file>")
    ap.add_argument("--post", default=None, help="processed vocal (default: *_AutoMIX_VocalBus.wav)")
    ap.add_argument("--at", nargs="*", default=[], help="analyse windows starting at these times (user notes say m:ss～) instead of sections")
    ap.add_argument("--win", type=float, default=1.0, help="window length in seconds for --at")
    ap.add_argument("--active-db", type=float, default=30.0,
                    help="frames quieter than (95th pct of pre - this) are ignored as silence")
    args = ap.parse_args()

    render_dir = resolve_render_dir(args.render)
    report = load_report(render_dir)
    pre, pre_name = _resolve_pre(args.pre, render_dir, report)
    post_path = Path(args.post) if args.post else first_glob(render_dir, "*_AutoMIX_VocalBus.wav")
    if post_path is None:
        raise SystemExit("post file not found; pass --post")
    post = load_ch_first(post_path)

    bands = dict(VOCAL_MIX_BANDS)
    bp_pre, tot_pre = _band_power_frames(pre.mean(axis=0), bands)
    bp_post, tot_post = _band_power_frames(post.mean(axis=0), bands)
    n = min(len(tot_pre), len(tot_post))
    bp_pre, tot_pre, bp_post, tot_post = bp_pre[:n], tot_pre[:n], bp_post[:n], tot_post[:n]
    eps = 1e-12
    pre_db = 10 * np.log10(tot_pre + eps)
    active = pre_db > (np.percentile(pre_db, 95) - args.active_db)

    if args.at:
        spans = []
        for t in args.at:
            c = parse_time(t)
            spans.append((fmt_time(c), c, c + args.win))
    else:
        spans = [(f"{s['index']:02d} {s['label']}", s["start_sec"], s["end_sec"]) for s in sections(report)]

    rows = []
    for name, t0, t1 in spans:
        f0, f1 = max(0, int(t0 * SR / HOP)), min(n, int(t1 * SR / HOP) + 1)
        sel = np.zeros(n, dtype=bool)
        sel[f0:f1] = True
        sel &= active
        if sel.sum() < 3:
            rows.append([name, f"{fmt_time(t0)}-{fmt_time(t1)}", "(silent)"] + [""] * len(bands))
            continue
        gain = 10 * np.log10((tot_post[sel].mean() + eps) / (tot_pre[sel].mean() + eps))
        bd = 10 * np.log10((bp_post[sel].mean(axis=0) + eps) / (bp_pre[sel].mean(axis=0) + eps)) - gain
        rows.append([name, f"{fmt_time(t0)}-{fmt_time(t1)}", f"{gain:+.1f}"] + [f"{v:+.1f}" for v in bd])

    print(f"render: {render_dir.name}  pre: {pre_name}  post: {post_path.name}  (band = Δband - Δgain, dB)")
    print_table(["span", "range", "gain"] + list(bands), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
