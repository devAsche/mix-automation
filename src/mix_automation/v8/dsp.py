"""v8 R3: side-chain style DSP done in numpy/scipy (pure functions).

Arrays: mono = (n,), stereo = (2, n) channels-first (same layout as io.audio_loader.load_audio).
Constants carry their source: rulebook `ID@m:ss` (docs/mix_knowledge/rulebook.md) or "R3 既定".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal

from mix_automation.v8.metrics._core import (
    HOP,
    N_FFT,
    P56_MARGIN_DB,
    frame_rms_db,
    p56_activity,
    phrases,
)

EPS = 1e-12

# Envelope/gain computers work on 1 ms blocks, then interpolate back to the sample grid.
# 1 ms is shorter than the fastest time constant used here (10 ms), so the block rate does not limit it.
_BLOCK_S = 0.001


def _asym_smooth(x: np.ndarray, coef_up: float, coef_down: float) -> np.ndarray:
    """One-pole smoother with separate rise/fall coefficients (coef = exp(-dt / tau))."""
    out = np.empty_like(x)
    s = float(x[0]) if len(x) else 0.0
    for i, v in enumerate(x.tolist()):
        c = coef_up if v > s else coef_down
        s = c * s + (1.0 - c) * v
        out[i] = s
    return out


def _block_view(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    blk = max(1, round(_BLOCK_S * sr))
    n_b = int(np.ceil(len(x) / blk))
    pad = np.pad(x.astype(np.float64), (0, n_b * blk - len(x)))
    return pad.reshape(n_b, blk), blk


def envelope(x: np.ndarray, sr: int, attack_ms: float, release_ms: float, mode: str = "peak") -> np.ndarray:
    """Per-sample dB envelope of a mono (or stereo -> max) signal, one-sided attack/release smoothing."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        a = np.max(np.abs(a), axis=0)
    blocks, blk = _block_view(a, sr)
    if mode == "peak":
        det = np.max(np.abs(blocks), axis=1)
    elif mode == "rms":
        det = np.sqrt(np.mean(blocks**2, axis=1))
    else:
        raise ValueError(f"unknown envelope mode {mode!r}")
    dt = blk / sr
    c_up = float(np.exp(-dt / max(attack_ms * 1e-3, 1e-6)))
    c_dn = float(np.exp(-dt / max(release_ms * 1e-3, 1e-6)))
    env = _asym_smooth(det, c_up, c_dn)
    env_db = 20.0 * np.log10(np.maximum(env, EPS))
    t_blk = (np.arange(len(env_db)) + 0.5) * blk
    return np.interp(np.arange(len(a)), t_blk, env_db)


def duck_gain(
    key: np.ndarray,
    sr: int,
    depth_db: float,
    thr_db: float,
    range_db: float,
    attack_ms: float = 20.0,
    release_ms: float = 250.0,
) -> np.ndarray:
    """Linear gain per sample: 0 dB below thr, falling linearly to -depth_db at thr + range_db."""
    if depth_db <= 0.0:
        return np.ones(np.asarray(key).shape[-1])
    env_db = envelope(key, sr, attack_ms, release_ms, mode="rms")
    amount = np.clip((env_db - thr_db) / max(range_db, 1e-6), 0.0, 1.0) * depth_db
    return 10.0 ** (-amount / 20.0)


# ---------------------------------------------------------------- multiband duck (RTVPdDmWrwo@41:11)

# Bands and weights: R3 既定 (design of R3; vocal intelligibility region 300 Hz - 6 kHz).
DUCK_BANDS_HZ: tuple[tuple[float, float], ...] = ((300.0, 1000.0), (1000.0, 3000.0), (3000.0, 6000.0))
DUCK_BAND_WEIGHTS: tuple[float, ...] = (0.5, 1.0, 0.8)
# R3 既定: no ducking when the vocal is >= 12 dB below the backing in that band, full depth at >= 0 dB.
DUCK_RATIO_FLOOR_DB = -12.0
DUCK_RATIO_FULL_DB = 0.0
# R3 既定: key counts as "singing" only above its own peak level - 50 dB.
DUCK_KEY_GATE_DB = 50.0
# Crossfade between adjacent bands and outer taper width, in octaves (R3 既定).
DUCK_XFADE_OCT = 1.0 / 3.0


def _band_weights(freqs: np.ndarray) -> np.ndarray:
    """(n_bands, n_bins) frequency weights; interior crossfades sum to 1, outer edges taper to 0."""
    lf = np.log2(np.maximum(freqs, 1.0))
    half = DUCK_XFADE_OCT / 2.0

    def rise(edge_hz: float, centred: bool) -> np.ndarray:
        e = np.log2(edge_hz)
        lo, hi = (e - half, e + half) if centred else (e - DUCK_XFADE_OCT, e)
        t = np.clip((lf - lo) / (hi - lo), 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * t)

    n = len(DUCK_BANDS_HZ)
    w = np.zeros((n, len(freqs)))
    for i, (lo_hz, hi_hz) in enumerate(DUCK_BANDS_HZ):
        up = rise(lo_hz, centred=i > 0)
        down = 1.0 - rise(hi_hz, centred=i < n - 1) if i < n - 1 else 1.0 - _outer_fall(lf, hi_hz)
        w[i] = up * down
    return w


def _outer_fall(lf: np.ndarray, edge_hz: float) -> np.ndarray:
    e = np.log2(edge_hz)
    t = np.clip((lf - e) / DUCK_XFADE_OCT, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _smooth_db_frames(gr_db: np.ndarray, frame_dt: float, attack_ms: float, release_ms: float) -> np.ndarray:
    """Smooth gain reduction (positive dB) per frame: rises with attack, falls with release."""
    c_up = float(np.exp(-frame_dt / (attack_ms * 1e-3)))
    c_dn = float(np.exp(-frame_dt / (release_ms * 1e-3)))
    return _asym_smooth(gr_db, c_up, c_dn)


_STFT_NFFT = 2048
_STFT_HOP = 512


def _sqrt_hann(n: int) -> np.ndarray:
    return np.sqrt(signal.get_window("hann", n, fftbins=True))


def multiband_duck(
    inst_stereo: np.ndarray,
    key_mono: np.ndarray,
    sr: int,
    max_db: float,
    attack_ms: float = 20.0,
    release_ms: float = 150.0,
) -> tuple[np.ndarray, dict]:
    """Band-wise ducking of the backing track keyed by the vocal (same gain on L and R)."""
    inst = np.asarray(inst_stereo, dtype=np.float64)
    n = inst.shape[1]
    key = np.zeros(n)
    k = np.asarray(key_mono, dtype=np.float64)[:n]
    key[: len(k)] = k
    stats: dict = {"max_db": float(max_db), "bands_hz": [list(b) for b in DUCK_BANDS_HZ], "weights": list(DUCK_BAND_WEIGHTS)}
    if max_db <= 0.0 or n < _STFT_NFFT:
        stats["bands"] = [{"mean_gr_db": 0.0, "max_gr_db": 0.0} for _ in DUCK_BANDS_HZ]
        return inst.astype(np.float32), stats

    win = _sqrt_hann(_STFT_NFFT)
    kw = {"fs": sr, "window": win, "nperseg": _STFT_NFFT, "noverlap": _STFT_NFFT - _STFT_HOP, "boundary": "zeros", "padded": True}
    freqs, _, z_i = signal.stft(inst, **kw)
    _, _, z_k = signal.stft(key, **kw)
    p_i = np.sum(np.abs(z_i) ** 2, axis=0)
    p_k = np.abs(z_k) ** 2
    frame_dt = _STFT_HOP / sr

    key_frame_db = 10.0 * np.log10(np.sum(p_k, axis=0) + EPS)
    key_on = key_frame_db > (np.max(key_frame_db) - DUCK_KEY_GATE_DB)

    w_f = _band_weights(freqs)
    gain_db_bins = np.zeros_like(p_i)
    band_stats = []
    for b, (lo_hz, hi_hz) in enumerate(DUCK_BANDS_HZ):
        sel = (freqs >= lo_hz) & (freqs < hi_hz)
        e_k = p_k[sel].sum(axis=0)
        e_i = p_i[sel].sum(axis=0) / 2.0  # per-channel average so a centred vocal compares like for like
        ratio_db = 10.0 * np.log10((e_k + EPS) / (e_i + EPS))
        t = np.clip((ratio_db - DUCK_RATIO_FLOOR_DB) / (DUCK_RATIO_FULL_DB - DUCK_RATIO_FLOOR_DB), 0.0, 1.0)
        gr = np.where(key_on & (e_k > EPS * 10), t * DUCK_BAND_WEIGHTS[b] * max_db, 0.0)
        gr = _smooth_db_frames(gr, frame_dt, attack_ms, release_ms)
        gain_db_bins -= np.outer(w_f[b], gr)
        on = key_on if np.any(key_on) else np.ones_like(key_on)
        band_stats.append({"mean_gr_db": round(float(np.mean(gr[on])), 3), "max_gr_db": round(float(np.max(gr)), 3)})

    g = 10.0 ** (gain_db_bins / 20.0)
    _, y = signal.istft(z_i * g[None, :, :], **{k2: v for k2, v in kw.items() if k2 not in ("boundary", "padded")}, boundary=True)
    out = y[:, :n]
    stats["bands"] = band_stats
    return out.astype(np.float32), stats


# ---------------------------------------------------------------- M/S compressor (RTVPdDmWrwo@41:31-43:25)


def ms_compress(
    stereo: np.ndarray,
    sr: int,
    thr_db: float,
    ratio: float = 2.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    target: str = "mid",
) -> tuple[np.ndarray, dict]:
    """Compress only Mid (or only Side); the other component is untouched."""
    x = np.asarray(stereo, dtype=np.float64)
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1])
    comp = mid if target == "mid" else side
    env_db = envelope(comp, sr, attack_ms, release_ms, mode="peak")
    gr = np.maximum(env_db - thr_db, 0.0) * (1.0 - 1.0 / ratio)
    comp = comp * 10.0 ** (-gr / 20.0)
    if target == "mid":
        mid = comp
    else:
        side = comp
    out = np.stack([mid + side, mid - side])
    loud = env_db > (np.max(env_db) - 60.0)  # R3 既定: GR stats over the non-silent part
    stats = {
        "thr_db": round(float(thr_db), 3),
        "ratio": ratio,
        "target": target,
        "mean_gr_db": round(float(np.mean(gr[loud])) if np.any(loud) else 0.0, 3),
        "max_gr_db": round(float(np.max(gr)), 3),
    }
    return out.astype(np.float32), stats


# ---------------------------------------------------------------- filters / doubler post


def butter_filter(x: np.ndarray, sr: int, kind: str, cutoff: float | tuple[float, float], order: int = 2) -> np.ndarray:
    sos = signal.butter(order, cutoff, btype=kind, fs=sr, output="sos")
    return signal.sosfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def doubler_post(stereo: np.ndarray, sr: int, center_db: float = -6.0, lpf_hz: float = 9000.0) -> np.ndarray:
    """Doubler return: Mid -6 dB and 8-10 kHz low-pass (JUXJShxfzKQ@12:36, 12:49)."""
    x = np.asarray(stereo, dtype=np.float64)
    mid = 0.5 * (x[0] + x[1]) * 10.0 ** (center_db / 20.0)
    side = 0.5 * (x[0] - x[1])
    y = np.stack([mid + side, mid - side])
    return butter_filter(y, sr, "lowpass", lpf_hz, order=2).astype(np.float32)


# ---------------------------------------------------------------- ping-pong delay (2afvdjOpVJQ@15:50)


def pingpong_delay(
    mono: np.ndarray,
    sr: int,
    delay_s: float,
    feedback: float = 0.3,
    hp: float = 400.0,
    lp: float = 4000.0,
    n_taps: int = 8,
) -> np.ndarray:
    """1/16 ping-pong: tap k at k*delay_s, alternating L (odd k) / R (even k); filter inside the feedback.

    Self-made DSP (kHs Dual Delay's parameter mapping was not verified for R3).
    """
    x = np.asarray(mono, dtype=np.float64)
    n = len(x)
    d = max(1, round(delay_s * sr))
    out = np.zeros((2, n))
    sos = signal.butter(2, (hp, lp), btype="bandpass", fs=sr, output="sos")
    y = x.copy()
    amp = 1.0
    for k in range(1, n_taps + 1):
        y = signal.sosfilt(sos, y)
        off = k * d
        if off >= n:
            break
        ch = 0 if k % 2 == 1 else 1
        out[ch, off:] += amp * y[: n - off]
        amp *= feedback
        if amp < 1e-3:  # -60 dB: inaudible under the mix
            break
    return out.astype(np.float32)


# ---------------------------------------------------------------- loudness helpers (BS.1770 K-weighting)


def _biquad_k_weighting(sr: int) -> np.ndarray:
    """ITU-R BS.1770 K-weighting as two RBJ biquads (same parameters as pyloudnorm)."""
    # Stage 1: high shelf +4 dB @ 1500 Hz, Q 1/sqrt(2); stage 2: high-pass 38 Hz, Q 0.5.
    def high_shelf(g_db: float, f0: float, q: float) -> np.ndarray:
        a = 10 ** (g_db / 40.0)
        w0 = 2 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2 * q)
        cw = np.cos(w0)
        b0 = a * ((a + 1) + (a - 1) * cw + 2 * np.sqrt(a) * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - 2 * np.sqrt(a) * alpha)
        a0 = (a + 1) - (a - 1) * cw + 2 * np.sqrt(a) * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - 2 * np.sqrt(a) * alpha
        return np.array([b0, b1, b2, a0, a1, a2]) / a0

    def high_pass(f0: float, q: float) -> np.ndarray:
        w0 = 2 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2 * q)
        cw = np.cos(w0)
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
        return np.concatenate([b, a]) / a[0]

    return np.stack([high_shelf(4.0, 1500.0, 1 / np.sqrt(2)), high_pass(38.0, 0.5)])


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    return signal.sosfilt(_biquad_k_weighting(sr), np.asarray(x, dtype=np.float64), axis=-1)


def _ms_sum(kx: np.ndarray, s0: int, s1: int) -> float:
    seg = kx[..., s0:s1]
    if seg.shape[-1] == 0:
        return 0.0
    ms = np.mean(seg**2, axis=-1)
    return float(np.sum(ms))


def lufs_ungated(ms_sum: float) -> float:
    return -0.691 + 10.0 * np.log10(max(ms_sum, EPS))


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """pyloudnorm integrated loudness; x mono (n,) or stereo (2, n)."""
    import pyloudnorm as pyln

    a = np.asarray(x, dtype=np.float64)
    data = a.T if a.ndim == 2 else a
    return float(pyln.Meter(sr).integrated_loudness(data))


# ---------------------------------------------------------------- phrase ride (-c0bHeO5ISs@4:39)


@dataclass
class RideResult:
    audio: np.ndarray
    global_gain_db: float
    phrases: list[dict] = field(default_factory=list)

    def stats(self) -> dict:
        gains = np.array([p["gain_db"] for p in self.phrases]) if self.phrases else np.zeros(1)
        return {
            "global_gain_db": round(self.global_gain_db, 3),
            "n_phrases": len(self.phrases),
            "n_corrected": int(np.sum(np.abs(gains) > 1e-6)),
            "max_abs_phrase_gain_db": round(float(np.max(np.abs(gains))), 3),
            "phrases": self.phrases,
        }


_RIDE_RAMP_S = 0.05  # 50 ms ramps at phrase boundaries (R3 既定)
_RIDE_BLOCK_S = 0.05  # 50 ms blocks ~ short-term loudness integration (Glasberg & Moore 2002, as v8.metrics._core)
_RIDE_REL_GATE_LU = -10.0  # ITU-R BS.1770-4 relative gate


def phrase_ride(
    lead_mono: np.ndarray,
    inst_stereo: np.ndarray,
    sr: int,
    target_lu: float,
    tol_lu: float,
    max_db: float = 6.0,
) -> RideResult:
    """Vocal-minus-backing loudness ride: one global gain, then per-phrase correction of the excess over tol.

    The lead is treated as dual mono (it sits centred in the stereo mix), so its BS.1770 loudness is the sum
    of two identical channels.
    """
    lead = np.asarray(lead_mono, dtype=np.float64)
    n = len(lead)
    inst = np.zeros((2, n))
    ii = np.asarray(inst_stereo, dtype=np.float64)[:, :n]
    inst[:, : ii.shape[1]] = ii
    k_lead = k_weight(lead, sr)
    k_inst = k_weight(inst, sr)

    segs, _, _ = phrases(lead.astype(np.float32), sr)
    if not segs:
        return RideResult(lead.astype(np.float32), 0.0, [])

    blk = max(1, round(_RIDE_BLOCK_S * sr))

    def gated_energy(s0: int, s1: int) -> tuple[float, float, int]:
        """(vocal ms, backing ms, n_samples) over the phrase's blocks that pass the BS.1770 relative gate
        (-10 LU below the phrase mean), so P.56 hangover tails do not dilute the phrase level."""
        nb = (s1 - s0) // blk
        if nb < 1:  # shorter than one block: a single block over the whole phrase
            nb, e1 = 1, s1
        else:
            e1 = s0 + nb * blk
        v = 2.0 * np.mean(k_lead[s0:e1].reshape(nb, -1) ** 2, axis=1)
        i = np.sum(np.mean(k_inst[:, s0:e1].reshape(2, nb, -1) ** 2, axis=2), axis=0)
        keep = v > np.mean(v) * 10 ** (_RIDE_REL_GATE_LU / 10.0)
        if not np.any(keep):
            keep = np.ones_like(v, dtype=bool)
        return float(np.mean(v[keep])), float(np.mean(i[keep])), int(np.sum(keep)) * (e1 - s0) // nb

    def d_lu(s0: int, s1: int) -> float:
        ev_, ei_, _ = gated_energy(s0, s1)
        return lufs_ungated(ev_) - lufs_ungated(ei_)

    # Global: gated energy over all phrases.
    idx = [(max(0, int(s.start * sr)), min(n, int(s.end * sr))) for s in segs]
    parts = [gated_energy(a, b) for a, b in idx if b - a >= 2]
    tot = sum(p[2] for p in parts)
    e_v = sum(p[0] * p[2] for p in parts) / tot
    e_i = sum(p[1] * p[2] for p in parts) / tot
    d_global = lufs_ungated(e_v) - lufs_ungated(e_i)
    g_global = float(np.clip(target_lu - d_global, -24.0, 24.0))

    gain_db = np.full(n, g_global)
    table = []
    ramp = max(1, round(_RIDE_RAMP_S * sr))
    for (a, b), s in zip(idx, segs, strict=True):
        if b - a < 2:
            continue
        d = d_lu(a, b) + g_global - target_lu
        corr = 0.0
        if abs(d) > tol_lu:
            corr = float(np.clip(-np.sign(d) * (abs(d) - tol_lu), -max_db, max_db))
        table.append({"start": round(s.start, 3), "end": round(s.end, 3), "d_lu": round(float(d), 3), "gain_db": round(corr, 3)})
        if corr == 0.0:
            continue
        # Trapezoid: ramp in over 50 ms before the phrase start, ramp out over 50 ms after its end.
        r0, r1 = max(0, a - ramp), min(n, b + ramp)
        env = np.ones(r1 - r0)
        lead_in = a - r0
        lead_out = r1 - b
        if lead_in > 0:
            env[:lead_in] = np.linspace(0.0, 1.0, lead_in, endpoint=False)
        if lead_out > 0:
            env[-lead_out:] = np.linspace(1.0, 0.0, lead_out, endpoint=True)
        gain_db[r0:r1] += corr * env
    out = lead * 10.0 ** (gain_db / 20.0)
    return RideResult(out.astype(np.float32), g_global, table)


# ---------------------------------------------------------------- calibration & measurement


@dataclass
class CalibResult:
    value: float
    measured: float
    target: float
    converged: bool
    history: list[tuple[float, float]]
    audio: Any = None

    def summary(self) -> dict:
        return {
            "value": round(self.value, 4),
            "measured": round(self.measured, 3),
            "target": round(self.target, 3),
            "converged": self.converged,
            "iterations": len(self.history),
            "history": [[round(p, 4), round(m, 3)] for p, m in self.history],
        }


def calibrate(
    render: Callable[[float], Any],
    measure: Callable[[Any], float],
    target: float,
    lo: float,
    hi: float,
    iters: int = 4,
    tol: float = 0.25,
) -> CalibResult:
    """Find param in [lo, hi] with measure(render(param)) ~= target for a monotonic response.

    Evaluates both ends, then up to `iters` Illinois (regula falsi) steps inside the bracket.
    If the target is outside the reachable range the nearer end is returned with converged=False.
    """
    hist: list[tuple[float, float]] = []
    best: dict[str, Any] = {}

    def ev(p: float) -> float:
        audio = render(p)
        m = float(measure(audio))
        hist.append((p, m))
        if not best or abs(m - target) < abs(best["m"] - target):
            best.update(p=p, m=m, audio=audio)  # keep only the best render in memory
        return m - target

    a, b = lo, hi
    fa, fb = ev(a), ev(b)
    if fa * fb < 0:
        for _ in range(iters):
            if min(abs(fa), abs(fb)) <= tol or fb == fa:
                break
            c = b - fb * (b - a) / (fb - fa)
            w = abs(b - a)  # safeguard: stay at least 10 % inside the bracket
            c = float(np.clip(c, min(a, b) + 0.1 * w, max(a, b) - 0.1 * w))
            fc = ev(c)
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5  # Illinois modification
            b, fb = c, fc
    return CalibResult(best["p"], best["m"], target, abs(best["m"] - target) <= tol, hist, best["audio"])


def active_mask(x: np.ndarray, sr: int) -> np.ndarray:
    """P.56 activity on the shared stft frame grid (v8.metrics._core)."""
    a = np.asarray(x, dtype=np.float64)
    mono = a.mean(axis=0) if a.ndim == 2 else a
    mask, _, _ = p56_activity(mono.astype(np.float32), sr)
    return mask


def gr_db(before: np.ndarray, after: np.ndarray, mask: np.ndarray, sr: int = 44100) -> float:
    """Mean per-frame level drop (dB) over the active frames; positive = reduction."""
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    b = b.mean(axis=0) if b.ndim == 2 else b
    a = a.mean(axis=0) if a.ndim == 2 else a
    n = min(len(a), len(b))
    _, lb = frame_rms_db(b[:n], sr, HOP, N_FFT)
    _, la = frame_rms_db(a[:n], sr, HOP, N_FFT)
    m = np.asarray(mask[: len(lb)], dtype=bool)
    if not np.any(m):
        return 0.0
    return float(np.mean(lb[m] - la[m]))


def p56_threshold_db(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(active_level_db, activity_threshold_db) of a mono signal; threshold = active level - P56 margin."""
    a = np.asarray(x, dtype=np.float64)
    mono = a.mean(axis=0) if a.ndim == 2 else a
    _, act, thr = p56_activity(mono.astype(np.float32), sr)
    return act, thr if np.isfinite(thr) else act - P56_MARGIN_DB


# ---------------------------------------------------------------- shared report metrics


def true_peak_dbfs(x: np.ndarray, oversample: int = 4) -> float:
    """True-peak approximation: max |x| after 4x polyphase oversampling (BS.1770-4 Annex 2 uses 4x)."""
    a = np.asarray(x, dtype=np.float64)
    up = signal.resample_poly(a, oversample, 1, axis=-1)
    return float(20.0 * np.log10(np.max(np.abs(up)) + EPS))


def mix_metrics(vocal_bus: np.ndarray, inst: np.ndarray, master: np.ndarray, sr: int) -> dict[str, float]:
    """voc_minus_inst_lu, vocal_side_mid_db (active region of the vocal bus), master_lufs."""
    vb = np.asarray(vocal_bus, dtype=np.float64)
    vb = np.stack([vb, vb]) if vb.ndim == 1 else vb
    lv = integrated_lufs(vb, sr)
    li = integrated_lufs(inst, sr)
    mid = 0.5 * (vb[0] + vb[1])
    side = 0.5 * (vb[0] - vb[1])
    _, pm = frame_rms_db(mid, sr, HOP, N_FFT)
    _, ps = frame_rms_db(side, sr, HOP, N_FFT)
    m = active_mask(mid, sr)[: len(pm)]
    e_m = np.sum(10 ** (pm[m] / 10.0)) if np.any(m) else EPS
    e_s = np.sum(10 ** (ps[m] / 10.0)) if np.any(m) else EPS
    return {
        "voc_minus_inst_lu": round(lv - li, 3),
        "vocal_side_mid_db": round(float(10.0 * np.log10((e_s + EPS) / (e_m + EPS))), 3),
        "master_lufs": round(integrated_lufs(master, sr), 3),
    }
