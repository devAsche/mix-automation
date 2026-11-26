"""F. Delay timing against the beat grid, and G. effect dropouts at note ends.

Beat grid: Beat This! (Foscarin, Schlueter & Widmer 2024, ISMIR; CPJKU/beat_this, MIT).

Timing tolerance: the just-noticeable displacement of one event in an isochronous sequence is about
2.5 % of the inter-onset interval for IOIs of ~250-1000 ms and about 6 ms below that
(Friberg & Sundberg 1995, JASA 98(5); summarised in Repp 2005, Psychon. Bull. Rev. 12(6)).
`timing_tolerance_s(ioi)` returns max(6 ms, 2.5 % IOI).

Dropout tolerance: loudness and timbre integrate over ~100-200 ms (Zwicker & Fastl, "Psychoacoustics",
sec. 8.4 temporal integration), so an effect that stops >= 0.1 s before the note ends leaves an
audible unprocessed tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mix_automation.v8.metrics._core import HOP, frame_rms_db, p56_activity, to_mono

JND_REL_IOI = 0.025
JND_MIN_S = 0.006
DROPOUT_MIN_S = 0.1

# Note values relative to one beat (quarter note) that a tempo delay is normally set to.
SUBDIVISIONS = {
    "1/2": 2.0,
    "1/4.": 1.5,
    "1/4": 1.0,
    "1/4T": 2.0 / 3.0,
    "1/8.": 0.75,
    "1/8": 0.5,
    "1/8T": 1.0 / 3.0,
    "1/16": 0.25,
}


def timing_tolerance_s(ioi_s: float) -> float:
    return max(JND_MIN_S, JND_REL_IOI * ioi_s)


def beat_grid(x: np.ndarray, sr: int, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(beats, downbeats) in seconds from Beat This! (checkpoint 'final0', minimal postprocessing)."""
    import torch
    from beat_this.inference import Audio2Beats

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    a2b = Audio2Beats(checkpoint_path="final0", device=dev, dbn=False)
    beats, downbeats = a2b(to_mono(x).astype(np.float32), sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def local_ioi(beats: np.ndarray, t: float) -> float:
    if len(beats) < 2:
        return float("nan")
    d = np.diff(beats)
    k = int(np.clip(np.searchsorted(beats, t) - 1, 0, len(d) - 1))
    lo, hi = max(0, k - 4), min(len(d), k + 5)
    return float(np.median(d[lo:hi]))


@dataclass
class DelayGridFit:
    delay_s: float
    beat_ioi_s: float
    best_note: str
    error_s: float  # delay minus nearest note value
    tolerance_s: float
    off_grid: bool


def delay_vs_grid(delay_s: float, beat_ioi_s: float) -> DelayGridFit:
    best = min(SUBDIVISIONS.items(), key=lambda kv: abs(delay_s - kv[1] * beat_ioi_s))
    err = delay_s - best[1] * beat_ioi_s
    tol = timing_tolerance_s(best[1] * beat_ioi_s)
    return DelayGridFit(delay_s, beat_ioi_s, best[0], err, tol, abs(err) > tol)


def echo_period(processed: np.ndarray, sr: int, t0: float, t1: float, min_s: float = 0.08, max_s: float = 0.8) -> float:
    """Dominant repetition period (s) of the processed vocal envelope in [t0, t1] (autocorrelation)."""
    m = to_mono(processed)
    seg = m[max(0, int(t0 * sr)) : min(len(m), int(t1 * sr))]
    if len(seg) < int(2 * max_s * sr / 2):
        return float("nan")
    hop = 64
    n = len(seg) // hop
    env = np.sqrt(np.mean(seg[: n * hop].reshape(n, hop) ** 2, axis=1))
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[n - 1 :]
    lo, hi = int(min_s * sr / hop), min(len(ac) - 1, int(max_s * sr / hop))
    if hi <= lo:
        return float("nan")
    k = lo + int(np.argmax(ac[lo:hi]))
    return k * hop / sr


@dataclass
class DropoutResult:
    fx_end: float
    note_end: float
    dropout_s: float  # > 0: the effect ends before the note does
    missing: bool


def note_end_after(pre: np.ndarray, sr: int, t: float, search_s: float = 1.5) -> float:
    """First time >= t - 0.25 s where the pre vocal falls below the P.56 activity threshold
    and stays below for at least one P.56 hangover (0.2 s)."""
    times, lev = frame_rms_db(pre, sr)
    _, _, thr = p56_activity(pre, sr)
    dt = HOP / sr
    hang = round(0.2 / dt)
    i0 = max(0, int(np.searchsorted(times, t - 0.25)))
    i1 = min(len(lev), int(np.searchsorted(times, t + search_s)))
    for i in range(i0, i1):
        if lev[i] < thr and np.all(lev[i : i + hang] < thr):
            return float(times[i])
    return float(times[i1 - 1]) if i1 > i0 else t


def fx_dropout(pre: np.ndarray, sr: int, fx_start: float, fx_end: float) -> DropoutResult:
    ne = note_end_after(pre, sr, fx_end)
    d = ne - fx_end
    return DropoutResult(fx_end, ne, d, d >= DROPOUT_MIN_S)
