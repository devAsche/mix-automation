"""B. Voice colour relative to the backing: thin_light, muddy, airy_fizzy.

Spectral shape is measured in ISO 266 octave bands on vocal-active frames, as vocal minus backing
level per band (dB) and as the vocal's own tilt. Indices are differences between band groups, so no
absolute level enters. Fizz is the change, from the pre-processing vocal to the processed vocal, of
the 8-16 kHz level relative to 1-4 kHz and of the 8-16 kHz spectral flatness (Wiener entropy;
Johnston 1988, IEEE JSAC 6(2)): high flatness = noise-like, non-harmonic content.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mix_automation.v8.metrics._core import (
    EPS,
    OCTAVE_CENTERS,
    band_power,
    db,
    octave_edges,
    stft_power,
)

# Band groups (octave centres, Hz). Body/low-mid 250-500, core/presence 1-4 k, air 8-16 k:
# the conventional vocal regions of e.g. Owsinski, "The Mixing Engineer's Handbook" (4th ed.),
# ch. 6 frequency chart; they are the same octave bands, grouped.
BODY = (250.0, 500.0)
CORE = (1000.0, 2000.0, 4000.0)
AIR = (8000.0, 16000.0)
TILT_RANGE = (250.0, 8000.0)


def _idx(centers: tuple[float, ...]) -> list[int]:
    return [OCTAVE_CENTERS.index(c) for c in centers]


def band_levels(p: np.ndarray, freqs: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Power-mean octave-band level (dB) over the selected frames."""
    sub = p[:, frames] if np.any(frames) else p[:, :0]
    out = []
    for fc in OCTAVE_CENTERS:
        bp = band_power(sub, freqs, *octave_edges(fc))
        out.append(float(db(np.mean(bp))) if bp.size else float("nan"))
    return np.array(out)


@dataclass
class ColourIndices:
    vi_bands_db: np.ndarray  # vocal minus backing per octave band
    vocal_bands_db: np.ndarray
    tilt_db_per_oct: float  # slope of the vocal's own band levels, 250 Hz - 8 kHz
    body_rel_db: float  # (V-I in 250-500) - (V-I in 1-4 k): low = thin/light
    air_rel_db: float  # (V-I in 8-16 k) - (V-I in 1-4 k): high = airy
    bright_rel_db: float  # (V-I in 4-16 k) - (V-I in 250-500): low = muddy/dark


def colour_from_levels(vocal_db: np.ndarray, backing_db: np.ndarray) -> ColourIndices:
    vi = vocal_db - backing_db
    body = float(np.nanmean(vi[_idx(BODY)]))
    core = float(np.nanmean(vi[_idx(CORE)]))
    air = float(np.nanmean(vi[_idx(AIR)]))
    hi = float(np.nanmean(vi[_idx((4000.0, 8000.0, 16000.0))]))
    sel = [i for i, c in enumerate(OCTAVE_CENTERS) if TILT_RANGE[0] <= c <= TILT_RANGE[1]]
    x = np.log2(np.array(OCTAVE_CENTERS)[sel])
    y = vocal_db[sel]
    ok = np.isfinite(y)
    tilt = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else float("nan")
    return ColourIndices(vi, vocal_db, tilt, body - core, air - core, hi - body)


def colour_indices(vocal: np.ndarray, backing: np.ndarray, sr: int, active: np.ndarray) -> ColourIndices:
    freqs, _, pv = stft_power(vocal, sr)
    _, _, pb = stft_power(backing, sr)
    n = min(pv.shape[1], pb.shape[1], len(active))
    act = active[:n]
    return colour_from_levels(band_levels(pv[:, :n], freqs, act), band_levels(pb[:, :n], freqs, act))


def colour_timeline(
    vocal: np.ndarray, backing: np.ndarray, sr: int, active: np.ndarray, times: np.ndarray, win_s: float, hop_s: float
) -> tuple[np.ndarray, list[ColourIndices | None]]:
    """Colour indices on sliding windows of active frames (None where the window has < 25% activity)."""
    freqs, _, pv = stft_power(vocal, sr)
    _, _, pb = stft_power(backing, sr)
    n = min(pv.shape[1], pb.shape[1], len(active), len(times))
    centers = np.arange(win_s / 2, times[n - 1], hop_s)
    out: list[ColourIndices | None] = []
    for c in centers:
        w = (times[:n] >= c - win_s / 2) & (times[:n] < c + win_s / 2)
        a = w & active[:n]
        if a.sum() < 0.25 * max(w.sum(), 1):
            out.append(None)
            continue
        out.append(colour_from_levels(band_levels(pv[:, :n], freqs, a), band_levels(pb[:, :n], freqs, a)))
    return centers, out


def hf_flatness(p: np.ndarray, freqs: np.ndarray, lo: float = 8000.0, hi: float = 16000.0) -> np.ndarray:
    sel = (freqs >= lo) & (freqs < hi)
    sub = p[sel] + EPS
    return np.exp(np.mean(np.log(sub), axis=0)) / np.mean(sub, axis=0)


@dataclass
class FizzIndices:
    hf_rel_delta_db: float  # change of (8-16 k minus 1-4 k) level, processed - pre
    hf_flatness_delta: float  # change of 8-16 k flatness on active frames, processed - pre
    hf_rel_proc_db: float
    hf_rel_pre_db: float


def fizz_indices(processed: np.ndarray, pre: np.ndarray, sr: int, active: np.ndarray) -> FizzIndices:
    freqs, _, pp = stft_power(processed, sr)
    _, _, pr = stft_power(pre, sr)
    n = min(pp.shape[1], pr.shape[1], len(active))
    a = active[:n]

    def rel(p: np.ndarray) -> float:
        hf = float(db(np.mean(band_power(p[:, :n][:, a], freqs, 8000.0, 16000.0))))
        mid = float(db(np.mean(band_power(p[:, :n][:, a], freqs, 1000.0, 4000.0))))
        return hf - mid

    rp, rr = rel(pp), rel(pr)
    fp = float(np.median(hf_flatness(pp[:, :n][:, a], freqs)))
    fr = float(np.median(hf_flatness(pr[:, :n][:, a], freqs)))
    return FizzIndices(rp - rr, fp - fr, rp, rr)
