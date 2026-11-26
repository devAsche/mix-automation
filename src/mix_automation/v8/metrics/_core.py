"""Shared building blocks for v8 metrics: framing, spectra, bands, voice activity, phrases.

Every function is pure: it takes arrays and a sample rate and returns arrays/scalars.
Constants carry their source in a comment; anything without a source is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12

# STFT used by all frame-based metrics. 2048 samples = 46 ms at 44.1 kHz, the same order as the
# ~50 ms integration window of short-term loudness (Glasberg & Moore 2002, JAES 50(5)).
N_FFT = 2048
HOP = 512

# ISO 266 preferred octave-band centre frequencies (Hz); band edges are fc / sqrt(2) .. fc * sqrt(2).
OCTAVE_CENTERS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)

# ITU-T P.56 (12/2011) active speech level, method B: envelope time constant 0.03 s,
# hangover 0.2 s, margin M = 15.9 dB between active level and activity threshold.
P56_HANGOVER_S = 0.2
P56_MARGIN_DB = 15.9
P56_TIME_CONSTANT_S = 0.03


def to_mono(x: np.ndarray) -> np.ndarray:
    """(n,), (ch, n) or (n, ch) -> float64 (n,). Channels are averaged."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.shape[0] <= 8 and a.shape[0] < a.shape[1]:
        return a.mean(axis=0)
    return a.mean(axis=1)


def to_stereo(x: np.ndarray) -> np.ndarray:
    """(n,), (ch, n) or (n, ch) -> float64 (2, n)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return np.stack([a, a])
    if a.shape[0] > a.shape[1]:
        a = a.T
    if a.shape[0] == 1:
        return np.concatenate([a, a])
    return a[:2]


def db(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, EPS))


def match_length(*arrays: np.ndarray) -> list[np.ndarray]:
    n = min(a.shape[-1] for a in arrays)
    return [a[..., :n] for a in arrays]


def stft_power(x: np.ndarray, sr: int, n_fft: int = N_FFT, hop: int = HOP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power spectrogram of a mono signal. Returns (freqs[F], times[T], P[F, T])."""
    m = to_mono(x)
    if len(m) < n_fft:
        m = np.pad(m, (0, n_fft - len(m)))
    n_frames = 1 + (len(m) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    win = np.hanning(n_fft)
    spec = np.fft.rfft(m[idx] * win, axis=1)
    # Parseval scaling: summing a frame over all bins gives the frame's mean-square value
    # (one-sided spectrum, so x2). A full-scale sine sums to 0.5 (-3 dBFS).
    p = (np.abs(spec) ** 2).T * (2.0 / (n_fft * np.sum(win**2)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = (np.arange(n_frames) * hop + n_fft / 2) / sr
    return freqs, times, p


def band_power(p: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (freqs >= lo) & (freqs < hi)
    if not np.any(sel):
        return np.zeros(p.shape[1])
    return p[sel].sum(axis=0)


def octave_edges(fc: float) -> tuple[float, float]:
    return fc / np.sqrt(2.0), fc * np.sqrt(2.0)


def octave_band_db(p: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """(len(OCTAVE_CENTERS), T) band levels in dB."""
    return np.stack([db(band_power(p, freqs, *octave_edges(fc))) for fc in OCTAVE_CENTERS])


def frame_rms_db(x: np.ndarray, sr: int, hop: int = HOP, n_fft: int = N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """Frame RMS level in dBFS on the same frame grid as stft_power."""
    m = to_mono(x)
    if len(m) < n_fft:
        m = np.pad(m, (0, n_fft - len(m)))
    n_frames = 1 + (len(m) - n_fft) // hop
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    starts = hop * np.arange(n_frames)
    e = (c[starts + n_fft] - c[starts]) / n_fft
    times = (starts + n_fft / 2) / sr
    return times, np.asarray(db(e))


def p56_activity(x: np.ndarray, sr: int, hop: int = HOP, n_fft: int = N_FFT) -> tuple[np.ndarray, float, float]:
    """Voice activity on the stft frame grid after ITU-T P.56 method B.

    Returns (active_mask[T], active_level_db, threshold_db). The threshold c is the level for which
    the mean energy of frames above c (+ hangover) sits P56_MARGIN_DB above c.
    """
    _times, lev = frame_rms_db(x, sr, hop, n_fft)
    frame_dt = hop / sr
    # One-pole smoothing with the P.56 time constant (applied to energy).
    a = np.exp(-frame_dt / P56_TIME_CONSTANT_S)
    e = 10 ** (lev / 10.0)
    from scipy.signal import lfilter

    env = lfilter([1 - a], [1, -a], e)
    env_db = np.asarray(db(env))
    hang = max(1, round(P56_HANGOVER_S / frame_dt))
    best = (np.inf, -np.inf, np.zeros(len(e), dtype=bool), -np.inf)
    top = float(np.max(env_db)) if len(env_db) else -120.0
    for c in np.arange(top, top - 100.0, -0.5):
        above = env_db >= c
        mask = _hangover(above, hang)
        if not np.any(mask):
            continue
        act = float(db(np.mean(e[mask])))
        gap = abs((act - c) - P56_MARGIN_DB)
        if gap < best[0]:
            best = (gap, act, mask, c)
        elif act - c > P56_MARGIN_DB + 3.0:
            break
    return best[2], float(best[1]), float(best[3])


def _hangover(above: np.ndarray, hang: int) -> np.ndarray:
    idx = np.arange(len(above))
    last = np.maximum.accumulate(np.where(above, idx, -(10**9)))
    return (idx - last) <= hang


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start


def mask_to_segments(mask: np.ndarray, times: np.ndarray, frame_dt: float) -> list[Segment]:
    segs: list[Segment] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            segs.append(Segment(float(times[i] - frame_dt / 2), float(times[j] + frame_dt / 2)))
            i = j + 1
        else:
            i += 1
    return segs


def phrases(x: np.ndarray, sr: int) -> tuple[list[Segment], np.ndarray, np.ndarray]:
    """Phrases = runs of the P.56 active mask (hangover bridges gaps < 0.2 s).

    Returns (segments, active_mask[T], times[T]) on the stft frame grid.
    """
    times, _ = frame_rms_db(x, sr)
    mask, _, _ = p56_activity(x, sr)
    return mask_to_segments(mask, times, HOP / sr), mask, times


def window_mask(times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (times >= t0) & (times < t1)


def percentile_rank(values: np.ndarray, v: float) -> float:
    """Share (0..100) of values strictly below v."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(v):
        return float("nan")
    return float(100.0 * np.mean(vals < v))
