"""H. Harmony prominence: partial loudness of a harmony layer against the lead, and its interval.

prominence_phon = loudness level of the harmony in the presence of (lead + backing) minus that of the
lead in the presence of (harmony + backing), both from the Moore et al. (1997) partial-loudness model
in loudness.py. 0 = as loud as the lead; the more negative, the more the harmony is tucked in.
interval_st: median of 12*log2(f0_harmony / f0_lead) over frames where both are voiced (pYIN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mix_automation.v8.metrics._core import EPS, Segment, p56_activity, to_mono, window_mask
from mix_automation.v8.metrics.loudness import partial_loudness_track, sone_ratio_to_phon


@dataclass
class HarmonyTrack:
    times: np.ndarray
    prominence_phon: np.ndarray  # per frame, NaN where the harmony is inaudible/absent
    harm_partial_sone: np.ndarray
    lead_partial_sone: np.ndarray


def harmony_prominence(
    harmony: np.ndarray,
    lead: np.ndarray,
    backing: np.ndarray,
    sr: int,
    lead_activity_ref: np.ndarray | None = None,
) -> HarmonyTrack:
    """`lead_activity_ref`: dry lead used only to decide where the lead is sung (the processed lead
    estimate carries reverb and residue, so its own P.56 activity is almost always on)."""
    h, ld, b = to_mono(harmony), to_mono(lead), to_mono(backing)
    n = min(len(h), len(ld), len(b))
    h, ld, b = h[:n], ld[:n], b[:n]
    th = partial_loudness_track(h, ld + b, sr)
    tl = partial_loudness_track(ld, h + b, sr)
    ratio = th.partial_sone / np.maximum(tl.partial_sone, EPS)
    prom = np.asarray(sone_ratio_to_phon(ratio), dtype=float)
    # Only frames where both parts are sung (P.56 activity of each stem on its own).
    ah, _, _ = p56_activity(h, sr)
    al, _, _ = p56_activity(ld if lead_activity_ref is None else to_mono(lead_activity_ref)[:n], sr)
    m = min(len(prom), len(ah), len(al))
    both = np.zeros(len(prom), dtype=bool)
    both[:m] = ah[:m] & al[:m]
    prom[~both] = np.nan
    return HarmonyTrack(th.times, prom, th.partial_sone, tl.partial_sone)


def f0_track(x: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    import librosa

    from mix_automation.v8.metrics._core import HOP, N_FFT

    target = 22050
    y = librosa.resample(to_mono(x).astype(np.float32), orig_sr=sr, target_sr=target)
    hop = round(HOP * target / sr)
    f0, vflag, _ = librosa.pyin(y, fmin=70.0, fmax=1400.0, sr=target, frame_length=N_FFT * target // sr, hop_length=hop)
    f0 = np.where(vflag, f0, np.nan)
    vt = np.arange(len(f0)) * hop / target
    idx = np.clip(np.searchsorted(vt, times), 0, len(f0) - 1)
    return f0[idx]


def interval_semitones(f0_harm: np.ndarray, f0_lead: np.ndarray, sel: np.ndarray | None = None) -> float:
    ok = np.isfinite(f0_harm) & np.isfinite(f0_lead) & (f0_harm > 0) & (f0_lead > 0)
    if sel is not None:
        ok &= sel
    if ok.sum() < 3:
        return float("nan")
    return float(np.median(12.0 * np.log2(f0_harm[ok] / f0_lead[ok])))


def window_stat(track: HarmonyTrack, seg: Segment) -> float:
    w = window_mask(track.times, seg.start, seg.end)
    v = track.prominence_phon[w]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 90)) if v.size else float("nan")
