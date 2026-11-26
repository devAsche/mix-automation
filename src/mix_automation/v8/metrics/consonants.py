"""C. Consonants: sibilance and plosive / word-initial attacks, detected acoustically.

Phoneme alignment arrives in Phase 4; here consonants are found from the pre-processing vocal
(or, for reference mixes without one, from the vocal stem itself):
- sibilant frames: pYIN unvoiced (voiced probability < 0.5; Mauch & Dixon 2014, ICASSP) and more
  than half of the energy above 4 kHz. Sibilant noise of /s/ /sh/ has its spectral peak at 3-8 kHz
  (Jongman, Wayland & Wong 2000, JASA 108(3)), so the 4-12 kHz share is its natural detector.
- attacks: spectral-flux onsets (librosa default onset detector, Boeck & Widmer 2013 superflux-style)
  preceded by unvoiced/inactive frames, i.e. word or syllable starts with a stop or a hard onset.
  Attack window 0-30 ms covers the burst plus the voice-onset time of voiceless stops (20-40 ms in
  Japanese; Riney, Takagi, Ota & Uchida 2007, J. Phonetics 35), body window 50-150 ms the vowel.

For each event three measures (as asked in the Phase 2 brief):
(a) delta_ratio_db: (consonant / following-vowel energy) processed minus the same ratio pre.
    Negative = crushed by processing, positive = emphasised.
(b) vi_db: vocal minus backing level in the consonant band (4-12 kHz) during the consonant.
(c) delta_crest_db: short-time peak over phrase RMS, processed minus pre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mix_automation.v8.metrics._core import (
    EPS,
    HOP,
    N_FFT,
    band_power,
    db,
    p56_activity,
    stft_power,
    to_mono,
)

SIB_BAND = (4000.0, 12000.0)
# Stop bursts spread their energy over the mid/high range (Stevens & Blumstein 1978, JASA 64(5)).
ATTACK_BAND = (1000.0, 8000.0)
SIB_SHARE = 0.5  # more than half of the (>100 Hz) energy in SIB_BAND
VOICED_PROB = 0.5
MIN_SIB_FRAMES = 3  # ~35 ms, below the shortest fricative durations in Jongman et al. (2000)
VOWEL_SEARCH_S = 0.15
VOWEL_LEN_S = 0.10
ATTACK_S = (0.0, 0.03)
BODY_S = (0.05, 0.15)


@dataclass
class ConsonantEvent:
    kind: str  # "sibilant" | "attack"
    start: float
    end: float
    vowel_start: float
    vowel_end: float


@dataclass
class ConsonantMeasure:
    event: ConsonantEvent
    ratio_pre_db: float
    ratio_proc_db: float
    delta_ratio_db: float
    vi_db: float
    delta_crest_db: float


def voicing_prob(x: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    """pYIN voiced probability interpolated onto `times` (librosa.pyin at 22.05 kHz)."""
    import librosa

    m = to_mono(x).astype(np.float32)
    target = 22050
    y = librosa.resample(m, orig_sr=sr, target_sr=target) if sr != target else m
    hop = round(HOP * target / sr)
    _, _, vp = librosa.pyin(
        y, fmin=70.0, fmax=1000.0, sr=target, frame_length=N_FFT * target // sr, hop_length=hop
    )
    vt = np.arange(len(vp)) * hop / target
    return np.interp(times, vt, np.nan_to_num(vp, nan=0.0))


def frame_features(x: np.ndarray, sr: int, vp: np.ndarray | None = None) -> dict[str, np.ndarray]:
    freqs, times, p = stft_power(x, sr)
    total = band_power(p, freqs, 100.0, sr / 2)
    sib = band_power(p, freqs, *SIB_BAND)
    active, _, _ = p56_activity(x, sr)
    n = min(len(times), len(active))
    if vp is None:
        vp = voicing_prob(x, sr, times[:n])
    return {
        "times": times[:n],
        "level_db": np.asarray(db(total[:n])),
        "sib_share": sib[:n] / (total[:n] + EPS),
        "voiced": vp[:n],
        "active": active[:n],
    }


def detect_consonants(x: np.ndarray, sr: int, feats: dict[str, np.ndarray] | None = None) -> list[ConsonantEvent]:
    import librosa

    f = feats or frame_features(x, sr)
    t, vp, act = f["times"], f["voiced"], f["active"]
    dt = HOP / sr
    n = len(t)
    events: list[ConsonantEvent] = []

    def vowel_after(i_end: int) -> tuple[float, float] | None:
        j_max = min(n, i_end + int(VOWEL_SEARCH_S / dt) + 1)
        for j in range(i_end, j_max):
            if vp[j] >= VOICED_PROB and act[j]:
                return float(t[j]), float(t[j] + VOWEL_LEN_S)
        return None

    sib = (vp < VOICED_PROB) & (f["sib_share"] > SIB_SHARE) & act
    i = 0
    while i < n:
        if sib[i]:
            j = i
            while j + 1 < n and sib[j + 1]:
                j += 1
            if j - i + 1 >= MIN_SIB_FRAMES:
                v = vowel_after(j + 1)
                if v:
                    events.append(ConsonantEvent("sibilant", float(t[i] - dt / 2), float(t[j] + dt / 2), *v))
            i = j + 1
        else:
            i += 1

    m = to_mono(x).astype(np.float32)
    onsets = librosa.onset.onset_detect(y=m, sr=sr, hop_length=HOP, units="time")
    for o in onsets:
        k = int(np.searchsorted(t, o))
        if k < 3 or k + 2 >= n:
            continue
        prior_unvoiced = bool(np.all((vp[k - 3 : k] < VOICED_PROB) | ~act[k - 3 : k]))
        if not prior_unvoiced or not act[min(k + 1, n - 1)]:
            continue
        o = float(o)
        events.append(ConsonantEvent("attack", o + ATTACK_S[0], o + ATTACK_S[1], o + BODY_S[0], o + BODY_S[1]))
    events.sort(key=lambda e: e.start)
    return events


def _seg_ms(x: np.ndarray, sr: int, a: float, b: float) -> float:
    i0, i1 = max(0, int(a * sr)), min(len(x), int(b * sr))
    if i1 <= i0:
        return EPS
    return float(np.mean(x[i0:i1] ** 2)) + EPS


def _seg_band_ms(x: np.ndarray, sr: int, a: float, b: float, lo: float, hi: float) -> float:
    i0, i1 = max(0, int(a * sr)), min(len(x), int(b * sr))
    seg = x[i0:i1]
    if len(seg) < 64:
        return EPS
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    return float(spec[(fr >= lo) & (fr < hi)].sum() / len(seg)) + EPS


def _peak(x: np.ndarray, sr: int, a: float, b: float) -> float:
    i0, i1 = max(0, int(a * sr)), min(len(x), int(b * sr))
    return float(np.max(np.abs(x[i0:i1]))) + EPS if i1 > i0 else EPS


def measure_consonants(
    events: list[ConsonantEvent],
    processed: np.ndarray,
    pre: np.ndarray | None,
    backing: np.ndarray | None,
    sr: int,
    phrase_rms_window_s: float = 1.0,
) -> list[ConsonantMeasure]:
    """Per-event measures. Without `pre`, deltas are NaN and ratio_proc is the absolute ratio."""
    xp = to_mono(processed)
    xr = to_mono(pre) if pre is not None else None
    xb = to_mono(backing) if backing is not None else None
    out: list[ConsonantMeasure] = []
    for e in events:
        rp = float(db(_seg_ms(xp, sr, e.start, e.end) / _seg_ms(xp, sr, e.vowel_start, e.vowel_end)))
        ctx = (e.start - phrase_rms_window_s / 2, e.end + phrase_rms_window_s / 2)
        crest_p = 20 * np.log10(_peak(xp, sr, e.start, e.end)) - float(db(_seg_ms(xp, sr, *ctx)))
        if xr is not None:
            rr = float(db(_seg_ms(xr, sr, e.start, e.end) / _seg_ms(xr, sr, e.vowel_start, e.vowel_end)))
            crest_r = 20 * np.log10(_peak(xr, sr, e.start, e.end)) - float(db(_seg_ms(xr, sr, *ctx)))
            d_ratio, d_crest = rp - rr, float(crest_p - crest_r)
        else:
            rr, d_ratio, d_crest = float("nan"), float("nan"), float("nan")
        if xb is not None:
            band = SIB_BAND if e.kind == "sibilant" else ATTACK_BAND
            vi = float(db(_seg_band_ms(xp, sr, e.start, e.end, *band) / _seg_band_ms(xb, sr, e.start, e.end, *band)))
        else:
            vi = float("nan")
        out.append(ConsonantMeasure(e, rr, rp, d_ratio, vi, d_crest))
    return out
