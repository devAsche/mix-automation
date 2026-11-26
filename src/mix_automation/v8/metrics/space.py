"""E. Space: vocal tail / decay compared with the backing's own reverberation, and vocal width.

- Vocal decay: at each phrase end the level of the vocal stem is followed from the last local
  maximum; the slope between -5 and -25 dB below it gives a T20-style decay time scaled to 60 dB
  (ISO 3382-1:2009 evaluation range for T20). A dry note release decays fast, added reverb/delay
  slows the decay. The same method is used for every signal (rendered stems and separated stems).
- Tail ratio: energy in the 0.5 s after the phrase end relative to the last 0.5 s of the phrase.
- Backing reverberation: blind maximum-likelihood decay-rate estimation of Ratnam et al. (2003,
  JASA 114(5)) as implemented in nuniz/blind_rt60 (MIT). It was developed on speech; on dense music
  it is only a relative indicator (see metric_validation.md).
- Width: vocal Side/Mid energy ratio (dB) on vocal-active frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mix_automation.v8.metrics._core import EPS, db, p56_activity, phrases, to_mono, to_stereo

DECAY_HOP = 256  # 5.8 ms at 44.1 kHz
DECAY_WIN = 512
DECAY_RANGE_DB = (5.0, 25.0)  # ISO 3382-1 T20 evaluation range
TAIL_S = 0.5
MIN_GAP_S = 0.3  # a phrase end needs at least this much room before the next phrase


@dataclass
class DecayStats:
    n_ends: int
    decay_t60_s: float  # median over phrase ends with a valid fit
    tail_ratio_db: float  # median tail energy re end-of-phrase energy


def _level_curve(x: np.ndarray) -> np.ndarray:
    m = to_mono(x)
    n = 1 + max(0, len(m) - DECAY_WIN) // DECAY_HOP
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    s = DECAY_HOP * np.arange(n)
    return np.asarray(db((c[np.minimum(s + DECAY_WIN, len(m))] - c[s]) / DECAY_WIN))


def phrase_end_decay(vocal: np.ndarray, sr: int, ref_for_phrases: np.ndarray | None = None) -> DecayStats:
    segs, _, _ = phrases(ref_for_phrases if ref_for_phrases is not None else vocal, sr)
    lev = _level_curve(vocal)
    dt = DECAY_HOP / sr
    m = to_mono(vocal)
    t60s, tails = [], []
    for k, s in enumerate(segs):
        nxt = segs[k + 1].start if k + 1 < len(segs) else len(m) / sr
        if nxt - s.end < MIN_GAP_S or s.dur < 0.3:
            continue
        # P.56 phrase end includes a 0.2 s hangover; search the last maximum in the final 0.5 s.
        i0 = max(0, int((s.end - 0.5) / dt))
        i1 = min(len(lev), int(s.end / dt))
        if i1 - i0 < 3:
            continue
        ipk = i0 + int(np.argmax(lev[i0:i1]))
        iend = min(len(lev), int(nxt / dt))
        rel = lev[ipk:iend] - lev[ipk]
        a = np.argmax(rel <= -DECAY_RANGE_DB[0]) if np.any(rel <= -DECAY_RANGE_DB[0]) else None
        b = np.argmax(rel <= -DECAY_RANGE_DB[1]) if np.any(rel <= -DECAY_RANGE_DB[1]) else None
        if a is not None and b is not None and b - a >= 2:
            x = np.arange(a, b + 1) * dt
            slope = np.polyfit(x, rel[a : b + 1], 1)[0]
            if slope < 0:
                t60s.append(-60.0 / slope)
        e_end = np.mean(m[max(0, int((s.end - TAIL_S) * sr)) : int(s.end * sr)] ** 2) + EPS
        e_tail = np.mean(m[int(s.end * sr) : int(min(s.end + TAIL_S, nxt) * sr)] ** 2) + EPS
        tails.append(float(db(e_tail / e_end)))
    return DecayStats(
        len(tails),
        float(np.median(t60s)) if t60s else float("nan"),
        float(np.median(tails)) if tails else float("nan"),
    )


def blind_rt60(x: np.ndarray, sr: int, fs_analysis: int = 8000, excerpt_s: float = 20.0) -> float:
    """Blind RT60 (s) of a mono mix, Ratnam et al. (2003) via nuniz/blind_rt60. NaN if unavailable.

    The estimator iterates per 200 ms frame and is slow (~1 min per 20 s of audio at 16 kHz), so it
    runs at the package's default 8 kHz with half-overlapping frames on three excerpts (25/50/75 %
    of the track) and returns the median of the three estimates.
    """
    try:
        import librosa
        from blind_rt60 import BlindRT60
    except ImportError:
        return float("nan")
    m = to_mono(x)
    ests = []
    for frac in (0.25, 0.5, 0.75):
        c = int(frac * len(m))
        h = int(excerpt_s * sr / 2)
        seg = m[max(0, c - h) : c + h]
        if len(seg) < sr:
            continue
        y = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=fs_analysis)
        try:
            est = BlindRT60(fs=fs_analysis, hop=0.1)
            ests.append(float(est.estimate(y.astype(np.float64), fs_analysis)))
        except Exception:  # noqa: BLE001, S112 - estimator fails on degenerate input; metric reports NaN
            continue
    ests = [e for e in ests if np.isfinite(e)]
    return float(np.median(ests)) if ests else float("nan")


def side_mid_db(vocal: np.ndarray, sr: int) -> float:
    st = to_stereo(vocal)
    act, _, _ = p56_activity(st.mean(axis=0), sr)
    mid = 0.5 * (st[0] + st[1])
    side = 0.5 * (st[0] - st[1])
    hop = 512
    n = min(len(act), len(mid) // hop)
    mask = np.repeat(act[:n], hop)
    return float(db(np.mean(side[: n * hop][mask] ** 2) / (np.mean(mid[: n * hop][mask] ** 2) + EPS)))
