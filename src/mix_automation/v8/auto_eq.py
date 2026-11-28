"""R5b: automatic starting point for the correction EQ (Persistent Q) and the Pultec air rule.

Correction EQ (`auto_profile`): resonance peaks of the uncorrected lead's long-term spectrum (voiced
frames only) become Peaking cuts, at most 1 in 250-700 Hz, 2 in 2-5 kHz and 2 in 8-14 kHz. Gains and
Q follow Ryo's own settings of R5 (エバ / L&D, profiles/correction_eq/): low-mid -3.5 dB Q 2.5, 2-5 kHz
-2 dB Q 3.5, 8-14 kHz -2 dB Q 3.3. No band adds level and there is no broad high cut.

Peak = local maximum of (fine-smoothed LTAS - wide-smoothed envelope) in dB, "prominence":
  250-700 Hz and 2-5 kHz: 1/12 oct over a 1 oct envelope (broad resonances; Ryo's cuts there are wide)
  8-14 kHz:              1/48 oct over a 1/3 oct envelope (the narrow "ジー"/"キー" peaks)
kept when prominence >= the range threshold, strongest first, >= 1/6 oct apart. Thresholds were set on
エバ and checked on L&D (docs/v8r_sections/README.md, cross-check table).

Broad high band (`broad_high_band`): the Pultec stays at the fixed R5 setting on every song. Ryo's net 5-16 kHz lift (Pultec + his EQ, re 1 kHz) was ~0 dB on エバ and
+1.5 dB on L&D, whose backing is brighter than the voice by more. The backing-minus-voice gap g in
6-16 kHz (1/3-oct shapes normalised over 200 Hz-10 kHz, voice = pre-tone lead, voiced frames) maps
linearly through those two songs to a net target clipped to 0..+2 dB; one broad Peaking band at
8.3 kHz Q 0.6 (the shape Ryo used on エバ) takes net target - Pultec (+1.76 dB), limited to -3..0 dB.
Provisional rule from two songs only.
"""

from __future__ import annotations

import numpy as np

N_FFT, HOP = 8192, 2048
VOICED_BELOW_P95_DB = 30.0
NORM_BAND = (200.0, 10000.0)
MIN_SEP_OCT = 1 / 6

# (lo Hz, hi Hz, max bands, fine oct, envelope oct, threshold dB, gain dB, Q)
RANGES: tuple[tuple[float, float, int, float, float, float, float, float], ...] = (
    (250.0, 700.0, 1, 1 / 12, 1.0, 1.0, -3.5, 2.5),
    (2000.0, 5000.0, 2, 1 / 12, 1.0, 1.0, -2.0, 3.5),
    (8000.0, 14000.0, 2, 1 / 48, 1 / 3, 0.5, -2.0, 3.3),
)

# UAD Pultec EQP-1A as configured in chain.lead_tone (100 CPS boost 2 / atten 3, 10 KCS, hf_q 4.5):
# hf boost -> mean response over 5-16 kHz re 1 kHz (dB), measured with noise
# (scripts/analysis/pultec_ceq_response.py, 2026-11-28). Even at boost 0 the plugin is +1.25 dB up there.
PULTEC_TABLE = ((0.0, 1.247), (0.5, 1.321), (1.0, 1.507), (1.5, 1.764), (2.0, 2.072), (2.5, 2.417),
                (3.0, 2.792), (4.0, 3.623))
# Rule points: g on the pre-tone lead vs the raw backing (エバ 修正後 stems, L&D) -> Ryo's net lift.
RULE_POINTS = ((5.10, 0.0), (8.07, 1.5))
NET_RANGE_DB = (0.0, 2.0)
PULTEC_NET_DB = 1.764  # the fixed R5 Pultec (boost 1.5 @ 10 KCS), PULTEC_TABLE
BROAD_HIGH_HZ, BROAD_HIGH_Q = 8300.0, 0.6
BROAD_HIGH_GAIN_DB = (-3.0, 0.0)


def _frames_power(x: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    fr = sliding_window_view(x, N_FFT)[::HOP]
    win = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((fr.shape[0], N_FFT // 2 + 1), dtype=np.float32)
    for i in range(0, fr.shape[0], 256):
        out[i:i + 256] = np.abs(np.fft.rfft(fr[i:i + 256] * win, axis=1)) ** 2
    return out


def _voiced(p: np.ndarray) -> np.ndarray:
    lev = 10 * np.log10(p.sum(axis=1) + 1e-20)
    return lev >= np.percentile(lev, 95) - VOICED_BELOW_P95_DB


def _smooth_db(power: np.ndarray, sr: int, grid: np.ndarray, width_oct: float) -> np.ndarray:
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    cs = np.concatenate([[0.0], np.cumsum(power, dtype=np.float64)])
    lo = np.searchsorted(f, grid * 2 ** (-width_oct / 2))
    hi = np.maximum(np.searchsorted(f, grid * 2 ** (width_oct / 2)), lo + 1)
    return 10 * np.log10((cs[hi] - cs[lo]) / (hi - lo) + 1e-20)


def _grid(per_oct: int = 96, lo: float = 80.0, hi: float = 16000.0) -> np.ndarray:
    return lo * 2 ** (np.arange(int(np.log2(hi / lo) * per_oct) + 1) / per_oct)


def voiced_ltas(x: np.ndarray) -> np.ndarray:
    p = _frames_power(x)
    return p[_voiced(p)].mean(axis=0)


def find_peaks(ltas: np.ndarray, sr: int) -> list[dict]:
    """Selected peaks per range: [{range, hz, prom_db, gain_db, q}] (strongest first within a range)."""
    from scipy.signal import find_peaks as _fp

    g = _grid()
    out = []
    for lo, hi, n_max, fine, env, thr, gain, q in RANGES:
        prom = _smooth_db(ltas, sr, g, fine) - _smooth_db(ltas, sr, g, env)
        pk, _ = _fp(prom)
        cand = sorted((k for k in pk if lo <= g[k] <= hi and prom[k] >= thr), key=lambda k: -prom[k])
        chosen: list[int] = []
        for k in cand:
            if len(chosen) >= n_max:
                break
            if all(abs(np.log2(g[k] / g[c])) >= MIN_SEP_OCT for c in chosen):
                chosen.append(k)
        out += [{"range": [lo, hi], "hz": float(g[k]), "prom_db": float(prom[k]), "gain_db": gain, "q": q}
                for k in chosen]
    return out


def auto_profile(x: np.ndarray, sr: int, voice: str | None = "ryo", gap_db: float | None = None) -> dict:
    """Correction EQ dict (correction_eq format, validated) from the uncorrected lead `x`.

    gap_db: high_gap_db(pre-tone lead, raw backing); adds the broad high band when its gain is < 0."""
    from mix_automation.v8 import correction_eq as ceq

    peaks = find_peaks(voiced_ltas(x), sr)
    high = broad_high_band(gap_db) if gap_db is not None else None
    if high is not None and high["gain_db"] < 0.0:
        peaks.append({"range": "broad_high", "hz": high["hz"], "prom_db": None, "gain_db": high["gain_db"],
                      "q": high["q"]})
    bands = [{"band": i + 1, "enabled": True, "type": "Peaking", "frequency_hz": round(p["hz"], 1),
              "gain_db": p["gain_db"], "q": p["q"], "slope_db_oct": None, "placement": "Stereo", "extra": {}}
             for i, p in enumerate(peaks)]
    src = {"auto": "v8.auto_eq.auto_profile", "broad_high": high, "peaks": [{k: (round(v, 2) if isinstance(v, float) else v)
                                                         for k, v in p.items()} for p in peaks]}
    return ceq.validate({"format": ceq.FORMAT, "voice": voice, "plugin": ceq.PLUGIN, "source": src,
                         "global": {}, "bands": bands})


def high_gap_db(voice: np.ndarray, inst: np.ndarray) -> float:
    """Backing minus voice, mean over 6-16 kHz of 1/3-oct shapes normalised over 200 Hz-10 kHz (dB).

    Voice: voiced frames only; backing: all frames. Both 44.1 kHz assumed by the rule points."""
    sr = 44100
    g = _grid(24)
    pv = _frames_power(voice)
    v = _smooth_db(pv[_voiced(pv)].mean(axis=0), sr, g, 1 / 3)
    i = _smooth_db(_frames_power(inst).mean(axis=0), sr, g, 1 / 3)
    m = (g >= NORM_BAND[0]) & (g <= NORM_BAND[1])
    gap = (i - i[m].mean()) - (v - v[m].mean())
    h = (g >= 6000.0) & (g < 16000.0)
    return float(gap[h].mean())


def broad_high_band(gap_db: float, fs: float = 48000.0) -> dict:
    """The broad high band (Peaking 8.3 kHz Q 0.6, the shape of Ryo's エバ band 9): its gain makes
    Pultec (fixed, +1.764 dB) + this band hit the net target of the gap rule; gain limited to -3..0 dB."""
    from mix_automation.v8.correction_eq import _biquad_db

    (g0, n0), (g1, n1) = RULE_POINTS
    target = float(np.clip(n0 + (gap_db - g0) * (n1 - n0) / (g1 - g0), *NET_RANGE_DB))
    f = np.geomspace(5000.0, 16000.0, 200)

    def net(gain: float) -> float:
        if gain == 0.0:
            return PULTEC_NET_DB
        r = _biquad_db("Peaking", BROAD_HIGH_HZ, gain, BROAD_HIGH_Q, f, fs)
        r1k = _biquad_db("Peaking", BROAD_HIGH_HZ, gain, BROAD_HIGH_Q, np.array([1000.0]), fs)[0]
        return PULTEC_NET_DB + float(np.mean(r - r1k))

    lo, hi = BROAD_HIGH_GAIN_DB
    if net(hi) <= target:
        gain = hi
    elif net(lo) >= target:
        gain = lo
    else:
        for _ in range(40):  # bisection: net() rises with gain
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if net(mid) < target else (lo, mid)
        gain = 0.5 * (lo + hi)
    return {"gap_db": round(gap_db, 3), "net_target_db": round(target, 3), "gain_db": round(gain, 2),
            "net_db": round(net(round(gain, 2)), 3), "hz": BROAD_HIGH_HZ, "q": BROAD_HIGH_Q}
