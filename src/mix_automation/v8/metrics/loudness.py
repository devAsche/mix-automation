"""A. Buried / sits-on-top: loudness and partial loudness of the vocal in the presence of the backing.

Model: Moore, Glasberg & Baer (1997) "A model for the prediction of thresholds, loudness, and
partial loudness", JAES 45(4), eqs. 17-18 (stationary). It is applied frame by frame (46 ms),
which is the usage of Ward & Reiss (2012, AES 133rd, "Loudness algorithms for automatic mixing")
for masking-aware mixing. MalcolmSlaney/LoudnessModel (BSD-2) was checked first: it implements
the time-varying loudness of the target alone but not target-plus-masker partial loudness, so the
partial-loudness part is implemented here from the paper's equations.

Simplifications (documented, not tuned):
- Auditory filters are the symmetric, level-independent roex(p) of Glasberg & Moore (1990,
  Hear. Res. 47), so upward spread of masking at high levels is not modelled.
- Outer/middle-ear transfer is taken as (absolute threshold) - (internal excitation at threshold),
  with the absolute threshold of Terhardt (1979, Hear. Res. 1) and the internal excitation E_THRQ
  of Moore et al. (1997) Fig. 3. This is the decomposition used by Moore et al. (1997, sec. II.B).
- The absolute playback level is an assumption (`listen_db_spl`); ratios and rankings change little
  with it, the absolute sone values do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mix_automation.v8.metrics._core import (
    EPS,
    Segment,
    band_power,
    db,
    stft_power,
    to_mono,
    window_mask,
)

# Moore, Glasberg & Baer (1997): compressive exponent and constants for G = 1 (f >= 500 Hz).
ALPHA_HI = 0.2
A_HI = 4.72  # value for G = 1 (0 dB)
# Loudness scaling constant C (ANSI S3.4-2007, which is the 1997 model with revised constants).
C_SONE = 0.046871
# Diotic presentation (both ears) = 2x the monaural loudness (ANSI S3.4-2007 sec. 3.6). With this a
# 1 kHz 40 dB SPL tone reads ~1.1 sone (tests/test_v8_metrics.py).
BINAURAL = 2.0
# Values read off Moore et al. (1997): K (dB) of the masked-threshold rule E_THRN = K*E_NOISE + E_THRQ
# (Fig. 9) and E_THRQ (dB, Fig. 3) at the frequencies below. Above 1 kHz K = -3 dB and above 500 Hz
# E_THRQ = 3.73 dB as stated in the paper. Tabulation as used by deeuu/loudness (kdB, internalExcitation).
_TAB_F = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000], dtype=float)
_TAB_K_DB = np.array([13.5, 10.04, 7.13, 4.92, 3.13, 1.54, 0.40, -0.51, -1.27, -1.88, -2.33, -2.67, -2.90, -3.0])
_TAB_ETHRQ_DB = np.array([28.25, 23.71, 19.33, 15.73, 12.68, 9.95, 8.01, 6.50, 5.30, 4.39, 3.73, 3.73, 3.73, 3.73])

# Listening level assumption: the vocal+backing mix is played back at this long-term level.
# 80 dB SPL is in the range used for music mixing/monitoring (79-85 dB SPL, e.g. SMPTE RP 200 /
# Katz, "Mastering Audio" 2015, ch. 15). Parameter, not tuned on the evaluations.
DEFAULT_LISTEN_DB_SPL = 80.0


def erb_hz(f: np.ndarray | float) -> np.ndarray | float:
    """Equivalent rectangular bandwidth, Glasberg & Moore (1990)."""
    return 24.7 * (4.37 * np.asarray(f) / 1000.0 + 1.0)


def erb_number(f: np.ndarray | float) -> np.ndarray | float:
    return 21.4 * np.log10(4.37 * np.asarray(f) / 1000.0 + 1.0)


def erb_number_to_hz(e: np.ndarray | float) -> np.ndarray | float:
    return (10 ** (np.asarray(e) / 21.4) - 1.0) * 1000.0 / 4.37


def erb_centers(fmin: float = 50.0, fmax: float = 15000.0, step: float = 0.5) -> np.ndarray:
    return np.asarray(erb_number_to_hz(np.arange(erb_number(fmin), erb_number(fmax), step)))


def terhardt_threshold_db(f: np.ndarray) -> np.ndarray:
    """Absolute threshold of hearing (dB SPL), Terhardt (1979)."""
    k = np.maximum(np.asarray(f, dtype=float), 20.0) / 1000.0
    return 3.64 * k**-0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k**4


def k_db(f: np.ndarray) -> np.ndarray:
    lf = np.log10(np.clip(f, _TAB_F[0], _TAB_F[-1]))
    return np.interp(lf, np.log10(_TAB_F), _TAB_K_DB)


def ethrq_db(f: np.ndarray) -> np.ndarray:
    lf = np.log10(np.clip(f, _TAB_F[0], _TAB_F[-1]))
    return np.interp(lf, np.log10(_TAB_F), _TAB_ETHRQ_DB)


def roex_weights(freqs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Symmetric roex(p) auditory filter weights [len(centers), len(freqs)], p = 4 fc / ERB(fc)."""
    fc = centers[:, None]
    p = 4.0 * fc / erb_hz(fc)
    g = np.abs(freqs[None, :] - fc) / fc
    w = (1.0 + p * g) * np.exp(-p * g)
    w[g > 0.8] = 0.0
    return w


@dataclass(frozen=True)
class LoudnessModel:
    """Precomputed filterbank and constants for one STFT grid."""

    freqs: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    ear_gain: np.ndarray  # linear power gain per FFT bin (outer/middle ear)
    ethrq: np.ndarray  # linear, per centre
    k: np.ndarray
    g: np.ndarray
    a: np.ndarray
    alpha: np.ndarray
    erb_step: float

    @classmethod
    def build(cls, freqs: np.ndarray, erb_step: float = 0.5) -> LoudnessModel:
        centers = erb_centers(step=erb_step)
        weights = roex_weights(freqs, centers)
        att_db = terhardt_threshold_db(freqs) - ethrq_db(freqs)
        ear_gain = 10 ** (-att_db / 10.0)
        ethrq_c = ethrq_db(centers)
        # Low-frequency compression (Moore et al. 1997, eqs. 4-5 and Fig. 4): G is the reduction of
        # cochlear amplifier gain relative to 500 Hz, expressed through E_THRQ; A and alpha follow G.
        g_db = ethrq_db(np.array([500.0]))[0] - ethrq_c
        g = 10 ** (g_db / 10.0)
        # A and alpha as functions of G (Moore et al. 1997, Fig. 5): A rises and alpha rises slightly
        # as G falls. Polynomial fits of that figure (same tabulation source as above).
        a = np.where(
            g_db >= 0,
            A_HI,
            -1.0706497e-6 * g_db**5 - 6.0648487e-5 * g_db**4 - 1.2047327e-3 * g_db**3
            - 6.8190418e-3 * g_db**2 - 0.11847826 * g_db + 4.7138722,
        )
        alpha = np.where(g_db >= 0, ALPHA_HI, 2.6864286e-5 * g_db**2 - 2.0023357e-3 * g_db + 0.19993107)
        return cls(freqs, centers, weights, ear_gain, 10 ** (ethrq_c / 10.0), 10 ** (k_db(centers) / 10.0), g, a, alpha, erb_step)

    def excitation(self, p_spl: np.ndarray) -> np.ndarray:
        """Power spectrum in SPL-linear units (1 = 0 dB SPL) [F, T] -> excitation [bands, T]."""
        return self.weights @ (p_spl * self.ear_gain[:, None])

    def specific_loudness(self, e: np.ndarray) -> np.ndarray:
        g, a, al = self.g[:, None], self.a[:, None], self.alpha[:, None]
        return C_SONE * ((g * e + a) ** al - a**al)

    def specific_partial_loudness(self, e_sig: np.ndarray, e_noise: np.ndarray) -> np.ndarray:
        """Moore et al. (1997) eqs. 17 (E_SIG >= E_THRN) and 18 (E_SIG < E_THRN)."""
        g, a, al = self.g[:, None], self.a[:, None], self.alpha[:, None]
        ethrq = self.ethrq[:, None]
        ethrn = self.k[:, None] * e_noise + ethrq
        e_tot = e_sig + e_noise
        hi = C_SONE * ((e_tot * g + a) ** al - a**al) - C_SONE * (
            (g * (e_noise + ethrn) + a) ** al - (ethrq * g + a) ** al
        ) * (ethrn / np.maximum(e_sig, EPS)) ** 0.3
        lo = C_SONE * (2.0 * e_sig / (e_sig + ethrn)) ** 1.5
        lo = lo * ((ethrq * g + a) ** al - a**al) / ((g * (e_noise + ethrn) + a) ** al - (g * e_noise + a) ** al)
        lo = lo * ((e_tot * g + a) ** al - (g * e_noise + a) ** al)
        out = np.where(e_sig >= ethrn, hi, lo)
        return np.maximum(out, 0.0)


@dataclass
class PartialLoudnessTrack:
    times: np.ndarray
    alone_sone: np.ndarray  # loudness of the vocal alone
    partial_sone: np.ndarray  # loudness of the vocal in the presence of the backing
    backing_sone: np.ndarray
    vi_1k4k_db: np.ndarray  # vocal minus backing level in 1-4 kHz (dB)


def spl_scale(vocal: np.ndarray, backing: np.ndarray, listen_db_spl: float) -> float:
    """Power factor mapping digital power to SPL-linear so that the mix RMS = listen_db_spl."""
    mix = to_mono(vocal) + to_mono(backing)
    ms = float(np.mean(mix**2)) + EPS
    return 10 ** (listen_db_spl / 10.0) / ms


def partial_loudness_track(
    vocal: np.ndarray,
    backing: np.ndarray,
    sr: int,
    listen_db_spl: float = DEFAULT_LISTEN_DB_SPL,
    model: LoudnessModel | None = None,
) -> PartialLoudnessTrack:
    v = to_mono(vocal)
    b = to_mono(backing)
    n = min(len(v), len(b))
    v, b = v[:n], b[:n]
    scale = spl_scale(v, b, listen_db_spl)
    freqs, times, pv = stft_power(v, sr)
    _, _, pb = stft_power(b, sr)
    pv = pv * scale
    pb = pb * scale
    m = model or LoudnessModel.build(freqs)
    ev = m.excitation(pv)
    eb = m.excitation(pb)
    step = m.erb_step
    k = step * BINAURAL
    alone = m.specific_loudness(ev).sum(axis=0) * k
    backing_l = m.specific_loudness(eb).sum(axis=0) * k
    partial = m.specific_partial_loudness(ev, eb).sum(axis=0) * k
    vi = db(band_power(pv, freqs, 1000.0, 4000.0)) - db(band_power(pb, freqs, 1000.0, 4000.0))
    return PartialLoudnessTrack(times, alone, partial, backing_l, np.asarray(vi))


def sone_ratio_to_phon(ratio: np.ndarray | float) -> np.ndarray | float:
    """Loudness ratio -> loudness-level difference: doubling of sones = +10 phon (Stevens' rule,
    valid above ~40 phon; Moore, 'An Introduction to the Psychology of Hearing', ch. 4)."""
    return 10.0 * np.log2(np.maximum(ratio, EPS))


@dataclass
class PhraseLoudness:
    seg: Segment
    alone_sone: float
    partial_sone: float
    masking_loss_phon: float  # partial vs alone; 0 = unmasked, more negative = more buried
    vi_1k4k_db: float


def phrase_loudness(track: PartialLoudnessTrack, segs: list[Segment]) -> list[PhraseLoudness]:
    out: list[PhraseLoudness] = []
    for s in segs:
        w = window_mask(track.times, s.start, s.end)
        if not np.any(w):
            continue
        a = float(np.mean(track.alone_sone[w]))
        p = float(np.mean(track.partial_sone[w]))
        out.append(PhraseLoudness(s, a, p, float(sone_ratio_to_phon(p / max(a, EPS))), float(np.median(track.vi_1k4k_db[w]))))
    return out


def masking_loss_frames(track: PartialLoudnessTrack, active: np.ndarray) -> np.ndarray:
    """Per-frame masking loss (phon) on active frames, NaN elsewhere."""
    loss = np.asarray(sone_ratio_to_phon(track.partial_sone / np.maximum(track.alone_sone, EPS)), dtype=float)
    loss[~active] = np.nan
    return loss


# EBU Tech 3341 "momentary" loudness window (400 ms). Used as the block length for masking loss so that
# rendered stems (dry phrases) and separated stems (wet, long P.56 segments) are compared on the same unit.
MOMENTARY_S = 0.4


def block_masking_loss(track: PartialLoudnessTrack, active: np.ndarray, block_s: float = MOMENTARY_S) -> np.ndarray:
    """Non-overlapping blocks with >= 50 % active frames -> array [[t_start, loss_phon, vi_1k4k_db], ...]."""
    t = track.times
    n = min(len(t), len(active))
    if n == 0:
        return np.zeros((0, 3))
    edges = np.arange(0.0, t[n - 1] + block_s, block_s)
    rows = []
    idx = np.searchsorted(t[:n], edges)
    for k in range(len(edges) - 1):
        a, b = idx[k], idx[k + 1]
        if b - a < 2:
            continue
        act = active[a:b]
        if act.mean() < 0.5:
            continue
        al = float(np.mean(track.alone_sone[a:b][act]))
        pa = float(np.mean(track.partial_sone[a:b][act]))
        rows.append([edges[k], float(sone_ratio_to_phon(pa / max(al, EPS))), float(np.median(track.vi_1k4k_db[a:b][act]))])
    return np.array(rows) if rows else np.zeros((0, 3))
