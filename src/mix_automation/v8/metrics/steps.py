"""D. Step changes: moments where the processing (processed minus pre) jumps.

Singing itself changes level and colour all the time, so every feature is the difference between
the processed vocal and the pre-processing vocal on the same frames. A step at time t is the mean of
that difference over the next `win_s` seconds minus the mean over the previous `win_s` seconds,
using only frames where the pre vocal is active.

Features (dB):
- level: frame level, processed - pre.
- bright: (2-8 kHz minus 150 Hz-1 kHz level), processed - pre. Darker = farther (high-frequency
  loss with distance; e.g. Zahorik, Brungart & Bronkhorst 2005, Acta Acustica 91, distance cues).
- wet: processed level in frames where the pre vocal is inactive minus processed level where it is
  active (a direct-to-reverberant proxy; more tail = farther, same source).
far_score = -level - bright + wet: positive = moved away from the listener.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mix_automation.v8.metrics._core import band_power, db, p56_activity, stft_power

# Intensity JND for broadband sounds ~0.5-1 dB (Zwicker & Fastl, "Psychoacoustics" 3rd ed., sec. 7.1);
# the upper end is used as the minimum step that counts as audible.
LEVEL_JND_DB = 1.0
DEFAULT_WIN_S = 3.0
DEFAULT_HOP_S = 0.25
MIN_ACTIVE_SHARE = 0.25


@dataclass
class StepTrack:
    times: np.ndarray
    d_level: np.ndarray
    d_bright: np.ndarray
    d_wet: np.ndarray
    far_score: np.ndarray


def _frame_feats(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, times, p = stft_power(x, sr)
    lev = np.asarray(db(band_power(p, freqs, 100.0, sr / 2)))
    br = np.asarray(db(band_power(p, freqs, 2000.0, 8000.0))) - np.asarray(db(band_power(p, freqs, 150.0, 1000.0)))
    return times, lev, br


def step_track(
    processed: np.ndarray, pre: np.ndarray, sr: int, win_s: float = DEFAULT_WIN_S, hop_s: float = DEFAULT_HOP_S
) -> StepTrack:
    tp, lp, bp = _frame_feats(processed, sr)
    _, lr, br = _frame_feats(pre, sr)
    act, _, _ = p56_activity(pre, sr)
    n = min(len(tp), len(lr), len(act))
    tp, lp, bp, lr, br, act = tp[:n], lp[:n], bp[:n], lr[:n], br[:n], act[:n]
    dl = lp - lr
    dbr = bp - br
    centers = np.arange(win_s, tp[-1] - win_s, hop_s) if n else np.array([])
    out = {k: np.full(len(centers), np.nan) for k in ("l", "b", "w")}

    def wet(sel: np.ndarray) -> float:
        on, off = sel & act, sel & ~act
        if on.sum() < 3 or off.sum() < 3:
            return np.nan
        return float(db(np.mean(10 ** (lp[off] / 10)))) - float(db(np.mean(10 ** (lp[on] / 10))))

    for i, c in enumerate(centers):
        before = (tp >= c - win_s) & (tp < c)
        after = (tp >= c) & (tp < c + win_s)
        ab, aa = before & act, after & act
        if ab.sum() < MIN_ACTIVE_SHARE * before.sum() or aa.sum() < MIN_ACTIVE_SHARE * after.sum():
            continue
        out["l"][i] = np.mean(dl[aa]) - np.mean(dl[ab])
        out["b"][i] = np.mean(dbr[aa]) - np.mean(dbr[ab])
        out["w"][i] = wet(after) - wet(before)
    far = -out["l"] - out["b"] + np.nan_to_num(out["w"], nan=0.0)
    return StepTrack(centers, out["l"], out["b"], out["w"], far)


def top_steps(track: StepTrack, n: int = 10, min_sep_s: float = 3.0, direction: str = "far") -> list[tuple[float, float]]:
    """Largest steps (time, score), separated by min_sep_s. direction 'far' or 'near'."""
    s = track.far_score if direction == "far" else -track.far_score
    order = np.argsort(-np.nan_to_num(s, nan=-np.inf))
    picked: list[tuple[float, float]] = []
    for i in order:
        if not np.isfinite(s[i]):
            break
        t = float(track.times[i])
        if all(abs(t - u) >= min_sep_s for u, _ in picked):
            picked.append((t, float(s[i])))
        if len(picked) >= n:
            break
    return picked
