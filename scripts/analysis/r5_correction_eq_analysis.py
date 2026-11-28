"""R5 correction EQ (Ryo's Persistent Q by ear) vs the backing track and the pro covers.

    python scripts/analysis/r5_correction_eq_analysis.py                 # every step, both songs
    python scripts/analysis/r5_correction_eq_analysis.py --steps 3 4     # some steps only

Steps (numbered as in the R5 analysis request, 2026-11-28):
  3  how much of the backing-minus-voice tilt g(f) = inst - voice the broad bands (Q < 2, static) r(f) and
     the whole EQ R(f) fill, per octave region. Shapes: 1/3-oct smoothed LTAS, mean over 200 Hz-10 kHz = 0.
     R is given twice: static RBJ of every unmuted band (dynamic/spectral bands at their static gain) and
     the real plugin's average effect (LTAS of lead_dry through Persistent Q / LTAS of lead_dry).
  4  narrow resonances of the uncorrected voice (1/48-oct LTAS minus its 1/3-oct envelope) in 2-5 kHz and
     8-14 kHz, top 3 each, matched against Ryo's narrow static cuts (Q >= 2) within 1/6 oct.
  5  per-section K-weighted V/I (compute_section_relative_profile, the metric of section_balance.py):
     v8 corrected vocal bus (lead_dry -> Persistent Q, plus every wet_* send; no extra comp) vs
     inst_ducked, and each pro cover's target_vocal / inst_pro over the same section mapped to cover time
     with match.orig_to_cover. Sections: the v7 render's sections (data/outputs/<song>_auto).
  6  エバ old vs 修正後 main stems: lengths, lag (cross-correlation), and which ones the loader keeps.

Outputs: short tables on stdout, JSON in data/outputs_v8r5/analysis/r5_correction_eq_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SR, fmt_time, print_table

ROOT = Path(__file__).resolve().parents[2]
RENDER = ROOT / "data" / "outputs_v8r5" / "render"
OUT_DIR = ROOT / "data" / "outputs_v8r5" / "analysis"
SONGS = {  # song key (match.SONG_REFS) -> (v8 render dir, correction EQ file, v7 render with sections)
    "エバ": ("エバ_v8", "ryo_エバ_20261128.json", "エバ_auto"),
    "ラブ&デストロイ": ("ラブandデストロイ_v8", "ryo_LD_20261128.json", "ラブandデストロイ_auto"),
}
N_FFT, HOP = 8192, 2048  # 5.4 Hz bins at 44.1 kHz (request: "FFT 8192")
VOICED_BELOW_P95_DB = 30.0  # frame counted as voiced if within 30 dB of the 95th-percentile frame level
NORM_BAND = (200.0, 10000.0)  # request: mean over 200 Hz-10 kHz set to 0
REGIONS = [("low", 100, 250), ("lowmid", 250, 800), ("mid", 800, 2500), ("himid", 2500, 6000), ("high", 6000, 16000)]
NO_DIFF_DB = 0.5  # request: |mean g| < 0.5 dB -> "no difference"
BROAD_Q = 2.0  # Ryo's split (request): broad shaping Q < 2, narrow cuts Q >= 2
PEAK_RANGES = [(2000.0, 5000.0), (8000.0, 14000.0)]
PEAK_MIN_PROM_DB = 0.5  # detection threshold on (1/48-oct LTAS - 1/3-oct envelope); reported in the table
PEAK_MIN_SEP_OCT = 1 / 6
MATCH_OCT = 1 / 6  # request: match within 1/6 oct


# ------------------------------------------------------------------ spectra


def _load(path: Path) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz, expected {SR}")
    return a.T


def _frames_power(x: np.ndarray) -> np.ndarray:
    """|STFT|^2, shape (frames, bins), Hann, N_FFT / HOP."""
    from numpy.lib.stride_tricks import sliding_window_view

    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    fr = sliding_window_view(x, N_FFT)[::HOP] * np.hanning(N_FFT).astype(np.float32)
    out = np.empty((fr.shape[0], N_FFT // 2 + 1), dtype=np.float32)
    for i in range(0, fr.shape[0], 512):
        out[i:i + 512] = np.abs(np.fft.rfft(fr[i:i + 512], axis=1)) ** 2
    return out


def voiced_mask(p: np.ndarray) -> np.ndarray:
    lev = 10 * np.log10(p.sum(axis=1) + 1e-20)
    return lev >= np.percentile(lev, 95) - VOICED_BELOW_P95_DB


def smooth_on_grid(power: np.ndarray, grid: np.ndarray, width_oct: float) -> np.ndarray:
    """Mean power over [f*2^-w/2, f*2^+w/2] for each grid frequency, in dB."""
    f = np.fft.rfftfreq(N_FFT, 1 / SR)
    cs = np.concatenate([[0.0], np.cumsum(power, dtype=np.float64)])
    lo = np.searchsorted(f, grid * 2 ** (-width_oct / 2))
    hi = np.maximum(np.searchsorted(f, grid * 2 ** (width_oct / 2)), lo + 1)
    return 10 * np.log10((cs[hi] - cs[lo]) / (hi - lo) + 1e-20)


def grid(per_oct: int, lo: float = 80.0, hi: float = 16000.0) -> np.ndarray:
    return lo * 2 ** (np.arange(int(np.log2(hi / lo) * per_oct) + 1) / per_oct)


def norm(db: np.ndarray, g: np.ndarray) -> np.ndarray:
    m = (g >= NORM_BAND[0]) & (g <= NORM_BAND[1])
    return db - db[m].mean()


# ------------------------------------------------------------------ correction EQ


def band_kind(b: dict) -> str:
    ex = b.get("extra") or {}
    if ex.get("mute"):
        return "muted"
    if ex.get("dynamic") or ex.get("spectral"):
        return "dyn"
    return "broad" if b["q"] < BROAD_Q else "narrow"


def static_db(bands: list[dict], f: np.ndarray) -> np.ndarray:
    from mix_automation.v8.correction_eq import RESPONSE_FS, _biquad_db

    out = np.zeros_like(f)
    for b in bands:
        if b["enabled"] and b["type"] in ("Peaking", "LowShelf", "HighShelf") and b["gain_db"] != 0.0:
            out += _biquad_db(b["type"], b["frequency_hz"], b["gain_db"], b["q"], f, RESPONSE_FS)
    return out


def apply_plugin(x: np.ndarray, cfg: dict) -> np.ndarray:
    """The pipeline's own path (chain.correction_eq): fresh Persistent Q, write_to_plugin, run."""
    from mix_automation.config.plugin_registry import get_plugin_map
    from mix_automation.v8 import chain
    from mix_automation.v8 import correction_eq as ceq

    spec = replace(get_plugin_map()["persistent_q"], key=chain.PERSISTENT_Q_CORRECTION_KEY)
    plug, _ = chain.load_stage_plugin(spec)
    ceq.write_to_plugin(plug, ceq.validate(cfg))
    return chain.run_plugin(plug, x, SR)


# ------------------------------------------------------------------ steps


def step3(song: str, d: dict) -> dict:
    g3 = grid(24)
    pv, pc, pi = d["P_dry"], d["P_corr"], d["P_inst"]
    vm = d["voiced"]
    v = smooth_on_grid(pv[vm].mean(0), g3, 1 / 3)
    i = smooth_on_grid(pi.mean(0), g3, 1 / 3)
    c = smooth_on_grid(pc[vm].mean(0), g3, 1 / 3)
    gap = norm(i, g3) - norm(v, g3)
    bands = d["cfg"]["bands"]
    r = norm(static_db([b for b in bands if band_kind(b) == "broad"], g3), g3)
    rs = norm(static_db([b for b in bands if band_kind(b) != "muted"], g3), g3)
    rp = norm(c - v, g3)
    rows, out = [], []
    for name, lo, hi in REGIONS:
        m = (g3 >= lo) & (g3 < hi)
        mg, mr, ms, mp = (float(a[m].mean()) for a in (gap, r, rs, rp))
        fill = (lambda x, mg=mg: "no gap" if abs(mg) < NO_DIFF_DB else f"{100 * x / mg:+.0f}%")
        rows.append([f"{name} {lo}-{hi}", f"{mg:+.1f}", f"{mr:+.1f}", f"{ms:+.1f}", f"{mp:+.1f}",
                     fill(mr), fill(ms), fill(mp)])
        out.append({"region": name, "lo": lo, "hi": hi, "g": mg, "r_broad": mr, "R_static": ms, "R_plugin": mp})
    print(f"\n[3] {song}: g = inst(mid) - voice; r = broad static; R = whole EQ (static RBJ / plugin measured)")
    print_table(["region Hz", "g", "r", "R_st", "R_pl", "r/g", "R_st/g", "R_pl/g"], rows)
    return {"regions": out}


def step4(song: str, d: dict) -> dict:
    from scipy.signal import find_peaks

    g48 = grid(48, 1000, 16000)
    spec = smooth_on_grid(d["P_dry"][d["voiced"]].mean(0), g48, 1 / 48)
    env = smooth_on_grid(d["P_dry"][d["voiced"]].mean(0), g48, 1 / 3)
    prom = spec - env
    cuts = [b for b in d["cfg"]["bands"] if band_kind(b) == "narrow" and b["gain_db"] < 0]
    rows, out = [], []
    for lo, hi in PEAK_RANGES:
        idx, _ = find_peaks(prom, height=PEAK_MIN_PROM_DB, distance=int(48 * PEAK_MIN_SEP_OCT))
        idx = [k for k in idx if lo <= g48[k] <= hi]
        top = sorted(idx, key=lambda k: -prom[k])[:3]
        rcut = [b for b in cuts if lo <= b["frequency_hz"] <= hi]
        for k in sorted(top, key=lambda k: g48[k]):
            f0 = float(g48[k])
            near = min(rcut, key=lambda b: abs(np.log2(b["frequency_hz"] / f0)), default=None)
            cents = 1200 * np.log2(near["frequency_hz"] / f0) if near else None
            hit = cents is not None and abs(cents) <= 1200 * MATCH_OCT
            rows.append([f"{lo / 1000:g}-{hi / 1000:g}k", f"{f0:.0f}", f"{prom[k]:+.1f}",
                         f"{near['frequency_hz']:.0f}" if near else "-",
                         f"{cents:+.0f}" if cents is not None else "-", "yes" if hit else "no"])
            out.append({"range": [lo, hi], "peak_hz": f0, "prom_db": float(prom[k]),
                        "nearest_cut_hz": near["frequency_hz"] if near else None, "cents": cents, "match": hit})
        for b in rcut:  # Ryo's cuts with no detected peak nearby
            if not any(abs(np.log2(b["frequency_hz"] / g48[k])) <= MATCH_OCT for k in top):
                k = int(np.argmin(np.abs(np.log2(g48 / b["frequency_hz"]))))
                rows.append([f"{lo / 1000:g}-{hi / 1000:g}k", "-", f"(at cut {prom[k]:+.1f})",
                             f"{b['frequency_hz']:.0f}", "-", "unmatched cut"])
    n_hit = sum(o["match"] for o in out)
    in_rng = [b for b in cuts if any(lo <= b["frequency_hz"] <= hi for lo, hi in PEAK_RANGES)]
    n_cut = sum(any(o["match"] and o["nearest_cut_hz"] == b["frequency_hz"] for o in out) for b in in_rng)
    print(f"\n[4] {song}: peaks = 1/48-oct LTAS - 1/3-oct envelope >= {PEAK_MIN_PROM_DB} dB, "
          f">= 1/6 oct apart, top 3 per range; match = Ryo's narrow cut within 1/6 oct (200 c)")
    print_table(["range", "peak Hz", "prom dB", "cut Hz", "cents", "match"], rows)
    print(f"    matched peaks {n_hit}/{len(out)}; Ryo's narrow cuts with a peak within 1/6 oct {n_cut}/{len(in_rng)} "
          f"{[round(b['frequency_hz']) for b in in_rng]}")
    return {"peaks": out, "threshold_db": PEAK_MIN_PROM_DB, "cuts_matched": n_cut, "cuts_in_ranges": len(in_rng)}


def v7_sections(v7_dir: str) -> list[tuple[str, float, float]]:
    rep = json.loads((ROOT / "data" / "outputs" / v7_dir / "mix_report.json").read_text(encoding="utf-8"))
    return [(s["label"], float(s["start_sec"]), float(s["end_sec"])) for s in rep["sections"]
            if "instrumental" not in s["label"].lower()]


def _kvi(voc: np.ndarray, inst: np.ndarray, t0: float, t1: float) -> float:
    from mix_automation.analysis.relative_profile import compute_section_relative_profile

    a, b = max(0, int(t0 * SR)), min(voc.shape[1], inst.shape[1], int(t1 * SR))
    if b - a < SR // 2:
        return float("nan")
    try:
        return float(compute_section_relative_profile(voc[:, a:b], inst[:, a:b], sr=SR).k_lufs_vi_ratio_db)
    except Exception:  # noqa: BLE001 - a silent section gives no value
        return float("nan")


def _st(a: np.ndarray) -> np.ndarray:
    return a if a.shape[0] == 2 else np.repeat(a[:1], 2, axis=0)


def step5(song: str, d: dict, write_wav: bool) -> dict:
    import soundfile as sf

    from mix_automation.v8.match import (
        SONG_REFS,
        TARGETS,
        cover_name,
        load_drift_curve,
        orig_to_cover,
    )

    stems = d["stems"]
    wets = sum(_load(p) for p in sorted(stems.glob("wet_*.wav")))
    bus = _st(d["corr"][None, :]) + wets
    bus0 = _st(d["dry"][None, :]) + wets
    inst = _st(_load(stems / "inst_ducked.wav"))
    if write_wav:  # for a cross-check with section_balance.py --vocal/--inst (data/ is git-ignored)
        sf.write(str(OUT_DIR / f"{d['render']}_vocalbus_corr.wav"), bus.T, SR, subtype="FLOAT")
    pros, align = {}, []
    for pair in SONG_REFS[song]:
        curve = load_drift_curve(pair)
        # time-axis check: v8 inst (10-40 s) vs the R2 inst_aligned (orig inst in cover time) at the mapped times
        ia = _load(TARGETS / pair / "inst_aligned.wav").mean(axis=0)
        c0 = int(orig_to_cover(10 * SR, curve))
        lag, pk = _xcorr_lag(inst.mean(axis=0)[10 * SR:40 * SR], ia[c0:c0 + 30 * SR], SR)
        align.append(f"{cover_name(pair)} {lag:+.0f} ms (r {pk:.2f})")
        pros[cover_name(pair)] = (curve, _st(_load(TARGETS / pair / "target_vocal.wav")),
                                  _st(_load(TARGETS / pair / "inst_pro.wav")))
    rows, out = [], []
    for label, t0, t1 in v7_sections(d["v7"]):
        v8c, v8d = _kvi(bus, inst, t0, t1), _kvi(bus0, inst, t0, t1)
        pk = {}
        for name, (curve, pv, pi) in pros.items():
            c0, c1 = (float(x) / SR for x in orig_to_cover(np.array([t0 * SR, t1 * SR]), curve))
            pk[name] = _kvi(pv, pi, c0, c1)
        med = float(np.nanmedian(list(pk.values()))) if any(np.isfinite(list(pk.values()))) else float("nan")
        rows.append([label, f"{fmt_time(t0)}-{fmt_time(t1)}", f"{v8d:+.1f}", f"{v8c:+.1f}",
                     *[f"{x:+.1f}" for x in pk.values()], f"{med:+.1f}", f"{v8c - med:+.1f}"])
        out.append({"section": label, "t0": t0, "t1": t1, "v8_pre": v8d, "v8_corr": v8c, "pro": pk,
                    "pro_median": med, "diff": v8c - med})
    print(f"\n[5] {song}: kVI (dB, K-weighted V/I); v8 = lead_dry (+corr EQ) + all sends vs inst_ducked; "
          f"pro = target_vocal / inst_pro at orig_to_cover times")
    print_table(["section", "time", "v8 pre", "v8 corr", *pros, "pro med", "corr-med"], rows)
    print("    time-axis check (lag of pro inst_aligned vs v8 inst at mapped time): " + "; ".join(align))
    return {"sections": out, "align": align}


def _xcorr_lag(a: np.ndarray, b: np.ndarray, sr: int, max_lag_s: float = 2.0) -> tuple[float, float]:
    """Lag of b relative to a (ms, positive = b later) and the normalised peak, on 4 kHz envelopes."""
    from scipy.signal import correlate, resample_poly

    a4, b4 = (resample_poly(x, 4000, sr) for x in (a, b))
    n = min(len(a4), len(b4))
    a4, b4 = a4[:n] - a4[:n].mean(), b4[:n] - b4[:n].mean()
    c = correlate(b4, a4, mode="full", method="fft")
    lags = np.arange(-n + 1, n)
    m = np.abs(lags) <= max_lag_s * 4000
    k = int(np.argmax(c * m))
    return 1000 * lags[k] / 4000, float(c[k] / (np.linalg.norm(a4) * np.linalg.norm(b4) + 1e-12))


def step6() -> dict:
    import soundfile as sf

    from mix_automation.io.audio_loader import (
        _classify_filename_role,
        _deduplicate_stem_records,
        load_audio,
    )
    from mix_automation.pipeline import discover_song_assets

    folder = ROOT / "data" / "raw" / "エバ"
    assets = discover_song_assets(folder)
    recs = []
    for p in sorted(assets["stems"]):
        st, _ = load_audio(p, target_sr=SR, mono=False)
        st = np.atleast_2d(st)
        recs.append({"path": p, "orig_channels": sf.info(str(p)).channels, "mtime": p.stat().st_mtime,
                     "stereo": st, "mono": np.mean(st, axis=0, dtype=np.float32), "role": _classify_filename_role(p.stem)})
    kept, skipped = _deduplicate_stem_records(recs, sr=SR)
    kept_names = {r["path"].name for r in kept}
    rows, out = [], []
    by = {r["path"].name: r for r in recs}
    for n in (1, 2, 3):
        old, new = by.get(f"エバ - edエバ - main {n}.wav"), by.get(f"エバ修正後 - edエバ - main {n}.wav")
        if old is None or new is None:
            continue
        lag, peak = _xcorr_lag(old["mono"], new["mono"], SR)
        cm = min(len(old["mono"]), len(new["mono"]))
        diff = new["mono"][:cm] - old["mono"][:cm]
        rel = 20 * np.log10(np.sqrt(np.mean(diff ** 2)) / (np.sqrt(np.mean(old["mono"][:cm] ** 2)) + 1e-12) + 1e-12)
        rows.append([f"main {n}", f"{len(old['mono']) / SR:.2f}", f"{len(new['mono']) / SR:.2f}",
                     f"{old['orig_channels']}/{new['orig_channels']}", f"{lag:+.1f}", f"{peak:.3f}", f"{rel:+.1f}",
                     "old" if old["path"].name in kept_names else "", "new" if new["path"].name in kept_names else ""])
        out.append({"main": n, "old_s": len(old["mono"]) / SR, "new_s": len(new["mono"]) / SR, "lag_ms": lag,
                    "xcorr_peak": peak, "diff_rel_db": rel, "old_kept": old["path"].name in kept_names,
                    "new_kept": new["path"].name in kept_names})
    print(f"\n[6] エバ: discover_song_assets stems = {len(assets['stems'])}; dedup kept "
          f"{sorted(kept_names)}; skipped {skipped}")
    print_table(["stem", "old s", "new s", "ch o/n", "lag ms", "xc peak", "diff/old dB", "kept", "kept"], rows)
    return {"n_stems": len(assets["stems"]), "kept": sorted(kept_names), "skipped": skipped, "pairs": out}


def prepare(song: str) -> dict:
    from mix_automation.v8 import correction_eq as ceq

    render, eq_file, v7 = SONGS[song]
    stems = RENDER / render / "stems"
    cfg = ceq.load(ceq.PROFILE_DIR / eq_file)
    dry = _load(stems / "lead_dry.wav")[0]
    corr = apply_plugin(dry, cfg)
    inst = _load(stems / "inst_ducked.wav")
    p_dry = _frames_power(dry)
    return {"render": render, "v7": v7, "stems": stems, "cfg": cfg, "dry": dry, "corr": corr,
            "P_dry": p_dry, "P_corr": _frames_power(corr), "P_inst": _frames_power(inst.mean(axis=0)),
            "voiced": voiced_mask(p_dry)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", nargs="*", type=int, default=[3, 4, 5, 6])
    ap.add_argument("--songs", nargs="*", default=list(SONGS))
    ap.add_argument("--write-wav", action="store_true", help="also write the corrected vocal bus (step 5)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    for song in args.songs:
        if not {3, 4, 5} & set(args.steps):
            break
        d = prepare(song)
        res[song] = {}
        if 3 in args.steps:
            res[song]["step3"] = step3(song, d)
        if 4 in args.steps:
            res[song]["step4"] = step4(song, d)
        if 5 in args.steps:
            res[song]["step5"] = step5(song, d, args.write_wav)
    if 6 in args.steps:
        res["step6"] = step6()
    out = OUT_DIR / "r5_correction_eq_analysis.json"
    if out.exists():  # a partial run (--steps / --songs) updates only its own entries
        prev = json.loads(out.read_text(encoding="utf-8"))
        for k, v in res.items():
            prev[k] = {**prev.get(k, {}), **v} if isinstance(v, dict) and k in SONGS else v
        res = prev
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\njson: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
