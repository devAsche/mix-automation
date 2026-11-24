"""v2 Multidimensional Relative Mix Profiling: Spectrum, Space (Tail/Width), Clarity & Archetypes.

Captures not only 6-band vocal-to-instrumental spectral margins and crest-factor glue,
but also:
  - `syllabic_clarity_index`: 2-8 Hz envelope modulation depth (detects muddy/reverberant smearing)
  - `wet_tail_ratio_db`: post-phrase (60-280ms) reverb/delay tail level relative to dry singing
  - `vocal_side_mid_db`: stereo width ratio of the full vocal bus (lead + harmonies + space)
  - `vocal_median_f0_hz`: fundamental frequency register (<295Hz low-mid vs >=295Hz high)
  - `backing_archetype`: automatic classification (`dense_rock_edm`, `groovy_mid_pop`, `intimate_ballad_sparse`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal

VOCAL_MIX_BANDS: dict[str, tuple[float, float]] = {
    "sub_low": (40.0, 130.0),          # Rumble / low collision
    "body_warmth": (160.0, 420.0),     # Vocal chest & core foundation
    "boxiness_mud": (450.0, 950.0),    # Boxy / honky muddiness zone
    "core_presence": (1500.0, 4200.0), # Forwardness, intimacy & stickiness
    "sibilance": (5000.0, 8500.0),     # Sibilant control zone
    "air_sheen": (8500.0, 16000.0),    # Silky top-end extension
}

EPS = 1e-10


@dataclass(frozen=True)
class SectionRelativeProfile:
    """Comprehensive relative acoustic & spatial metrics between vocal bus and backing track."""

    vocal_rms_db: float
    inst_rms_db: float
    broadband_vi_ratio_db: float
    vocal_crest_db: float
    inst_crest_db: float
    crest_difference_db: float
    position_stability_std_db: float
    harsh_resonance_index_db: float
    syllabic_clarity_index: float
    wet_tail_ratio_db: float
    vocal_side_mid_db: float
    backing_archetype: str
    band_margins_db: dict[str, float]
    inst_band_levels_db: dict[str, float]
    active_frame_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SectionTransferAdjustment:
    """Calculated parameter offsets to carry an anchor mix balance to another section."""

    input_gain_delta_db: float
    compressor_threshold_delta_db: float
    eq_band_deltas_db: dict[str, float]
    dynamic_unmask_delta_db: float


def _to_mono_and_side(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return `(mono_mid, side_or_none)` from 1D or 2D audio array."""
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False), None
    if audio.shape[0] <= 2 and audio.shape[0] < audio.shape[1]:
        if audio.shape[0] == 1:
            return audio[0].astype(np.float64), None
        mid = 0.5 * (audio[0].astype(np.float64) + audio[1].astype(np.float64))
        side = 0.5 * (audio[0].astype(np.float64) - audio[1].astype(np.float64))
        return mid, side
    if audio.shape[1] == 1:
        return audio[:, 0].astype(np.float64), None
    mid = 0.5 * (audio[:, 0].astype(np.float64) + audio[:, 1].astype(np.float64))
    side = 0.5 * (audio[:, 0].astype(np.float64) - audio[:, 1].astype(np.float64))
    return mid, side


def _db(power_or_amp: float | np.ndarray, is_power: bool = True) -> float | np.ndarray:
    scale = 10.0 if is_power else 20.0
    return scale * np.log10(np.maximum(power_or_amp, EPS))


def classify_backing_archetype(inst_crest_db: float, inst_band_levels_db: dict[str, float]) -> str:
    """Classify backing track into one of 3 production archetypes for plugin chain selection."""
    mid_hi_density = inst_band_levels_db["core_presence"] - inst_band_levels_db["body_warmth"]
    if inst_crest_db >= 11.4 and mid_hi_density < -3.2:
        return "intimate_ballad_sparse"
    if inst_crest_db <= 10.2 or mid_hi_density > -2.4:
        return "dense_rock_edm"
    return "groovy_mid_pop"


def _compute_syllabic_clarity_and_tail(
    v_frame_power: np.ndarray,
    active_mask: np.ndarray,
    hop_sec: float,
) -> tuple[float, float]:
    """Measure 2-8 Hz syllabic modulation depth (clarity) and post-phrase wet tail decay ratio."""
    env_amp = np.sqrt(np.maximum(v_frame_power, EPS))
    fs_env = 1.0 / hop_sec  # ~40 Hz

    # 1. Syllabic Clarity Index: normalized 2-8 Hz modulation energy over active regions
    if len(env_amp) >= 32 and fs_env > 18.0:
        sos_mod = signal.butter(2, [2.0, 8.0], btype="bandpass", fs=fs_env, output="sos")
        mod_sig = signal.sosfiltfilt(sos_mod, env_amp)
        act_mean = float(np.mean(env_amp[active_mask])) + 1e-9
        mod_rms = float(np.sqrt(np.mean(mod_sig[active_mask] ** 2)))
        clarity = round(float(mod_rms / act_mean), 4)
    else:
        clarity = 0.35

    # 2. Post-phrase Wet Tail Ratio (dB): energy 60-280ms immediately after active phrase ends
    labels, n_reg = ndimage.label(active_mask)
    tail_ratios: list[float] = []
    f_start_off = max(2, int(0.06 / hop_sec))
    f_end_off = max(f_start_off + 2, int(0.28 / hop_sec))

    for r_i in range(1, n_reg + 1):
        idxs = np.where(labels == r_i)[0]
        if len(idxs) * hop_sec < 0.25:
            continue
        end_f = idxs[-1]
        t0 = end_f + f_start_off
        t1 = min(len(v_frame_power), end_f + f_end_off)
        if t1 - t0 >= 2 and not np.any(active_mask[t0:t1]):
            phrase_p = float(np.mean(v_frame_power[idxs[-min(len(idxs), 10) :]])) + EPS
            tail_p = float(np.mean(v_frame_power[t0:t1])) + EPS
            tail_ratios.append(10.0 * np.log10(tail_p / phrase_p))

    wet_tail_db = round(float(np.median(tail_ratios)), 2) if tail_ratios else -22.0
    return clarity, wet_tail_db


def compute_section_relative_profile(
    vocal: np.ndarray,
    instrumental: np.ndarray,
    sr: int,
    frame_sec: float = 0.05,
    hop_sec: float = 0.025,
    activity_rel_threshold_db: float = -26.0,
) -> SectionRelativeProfile:
    """Extract multidimensional relative vocal-to-instrumental acoustic & spatial profile."""
    v_mono, v_side = _to_mono_and_side(vocal)
    i_mono, _ = _to_mono_and_side(instrumental)
    length = min(len(v_mono), len(i_mono))
    if length == 0:
        raise ValueError("Audio inputs must not be empty.")
    v_mono = v_mono[:length]
    i_mono = i_mono[:length]
    if v_side is not None:
        v_side = v_side[:length]

    n_fft = int(2 ** np.ceil(np.log2(max(int(sr * frame_sec), 1024))))
    hop_length = max(int(sr * hop_sec), 256)
    actual_hop_sec = hop_length / sr

    freqs, _, v_stft = signal.stft(v_mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, i_stft = signal.stft(i_mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    v_pow = np.abs(v_stft) ** 2
    i_pow = np.abs(i_stft) ** 2

    v_frame_power = np.sum(v_pow, axis=0)
    peak_frame_db = float(_db(np.percentile(v_frame_power, 95), is_power=True))
    v_frame_db = _db(v_frame_power, is_power=True)
    active_mask = v_frame_db >= (peak_frame_db + activity_rel_threshold_db)
    if not np.any(active_mask):
        active_mask = np.ones_like(v_frame_power, dtype=bool)

    active_ratio = float(np.mean(active_mask))
    i_frame_power = np.sum(i_pow, axis=0)

    v_rms_active_db = float(_db(np.mean(v_frame_power[active_mask]), is_power=True))
    i_rms_active_db = float(_db(np.mean(i_frame_power[active_mask]), is_power=True))
    broadband_vi_db = v_rms_active_db - i_rms_active_db

    frame_vi_diff_db = v_frame_db[active_mask] - _db(i_frame_power[active_mask], is_power=True)
    position_stability_std_db = float(np.std(frame_vi_diff_db))

    v_peak_db = float(_db(np.percentile(np.abs(v_mono), 99.5), is_power=False))
    i_peak_db = float(_db(np.percentile(np.abs(i_mono), 99.5), is_power=False))
    v_time_rms_db = float(
        _db(np.sqrt(np.mean(v_mono**2) + EPS) / max(np.sqrt(active_ratio), 0.1), is_power=False)
    )
    i_time_rms_db = float(_db(np.sqrt(np.mean(i_mono**2) + EPS), is_power=False))
    v_crest_db = max(v_peak_db - v_time_rms_db, 0.0)
    i_crest_db = max(i_peak_db - i_time_rms_db, 0.0)

    band_margins_db: dict[str, float] = {}
    inst_band_levels_db: dict[str, float] = {}
    for band_name, (f_lo, f_hi) in VOCAL_MIX_BANDS.items():
        bin_mask = (freqs >= f_lo) & (freqs < f_hi)
        v_band_mean = float(np.mean(np.sum(v_pow[bin_mask][:, active_mask], axis=0)))
        i_band_mean = float(np.mean(np.sum(i_pow[bin_mask][:, active_mask], axis=0)))
        v_b_db = float(_db(v_band_mean, is_power=True))
        i_b_db = float(_db(i_band_mean, is_power=True))
        band_margins_db[band_name] = round(v_b_db - i_b_db, 3)
        inst_band_levels_db[band_name] = round(i_b_db, 3)

    harsh_mask = (freqs >= 2000.0) & (freqs <= 5000.0)
    v_harsh_spec_db = _db(np.mean(v_pow[harsh_mask][:, active_mask], axis=1), is_power=True)
    smooth_width = max(5, len(v_harsh_spec_db) // 8)
    kernel = np.ones(smooth_width) / smooth_width
    smoothed_env_db = np.convolve(v_harsh_spec_db, kernel, mode="same")
    prominence_db = np.maximum(v_harsh_spec_db - smoothed_env_db, 0.0)
    harsh_resonance_index_db = float(np.percentile(prominence_db, 90))

    clarity_idx, wet_tail_db = _compute_syllabic_clarity_and_tail(
        v_frame_power, active_mask, hop_sec=actual_hop_sec
    )

    if v_side is not None:
        mid_p = float(np.mean(v_mono**2)) + EPS
        side_p = float(np.mean(v_side**2)) + EPS
        side_mid_db = round(float(10.0 * np.log10(side_p / mid_p)), 2)
    else:
        side_mid_db = -30.0

    archetype = classify_backing_archetype(i_crest_db, inst_band_levels_db)

    return SectionRelativeProfile(
        vocal_rms_db=round(v_rms_active_db, 3),
        inst_rms_db=round(i_rms_active_db, 3),
        broadband_vi_ratio_db=round(broadband_vi_db, 3),
        vocal_crest_db=round(v_crest_db, 3),
        inst_crest_db=round(i_crest_db, 3),
        crest_difference_db=round(v_crest_db - i_crest_db, 3),
        position_stability_std_db=round(position_stability_std_db, 3),
        harsh_resonance_index_db=round(harsh_resonance_index_db, 3),
        syllabic_clarity_index=clarity_idx,
        wet_tail_ratio_db=wet_tail_db,
        vocal_side_mid_db=side_mid_db,
        backing_archetype=archetype,
        band_margins_db=band_margins_db,
        inst_band_levels_db=inst_band_levels_db,
        active_frame_ratio=round(active_ratio, 3),
    )


def compute_anchor_propagation(
    anchor_profile: SectionRelativeProfile,
    target_raw_profile: SectionRelativeProfile,
    eq_clamp_db: float = 4.0,
) -> SectionTransferAdjustment:
    """Translate the approved mix balance from an anchor section to another section."""
    gain_delta_db = anchor_profile.broadband_vi_ratio_db - target_raw_profile.broadband_vi_ratio_db

    eq_deltas: dict[str, float] = {}
    for band_name in VOCAL_MIX_BANDS:
        projected_margin = target_raw_profile.band_margins_db[band_name] + gain_delta_db
        needed_eq = anchor_profile.band_margins_db[band_name] - projected_margin
        weight = 0.35 if band_name in ("sub_low", "sibilance") else 0.70
        eq_deltas[band_name] = round(float(np.clip(needed_eq * weight, -eq_clamp_db, eq_clamp_db)), 2)

    crest_gap = target_raw_profile.crest_difference_db - anchor_profile.crest_difference_db
    comp_threshold_delta_db = round(float(np.clip(-0.60 * crest_gap, -5.0, 5.0)), 2)

    presence_inst_jump = (
        target_raw_profile.inst_band_levels_db["core_presence"]
        - anchor_profile.inst_band_levels_db["core_presence"]
    )
    dynamic_unmask_delta_db = round(float(np.clip(0.35 * presence_inst_jump, -2.0, 2.5)), 2)

    return SectionTransferAdjustment(
        input_gain_delta_db=round(float(gain_delta_db), 2),
        compressor_threshold_delta_db=comp_threshold_delta_db,
        eq_band_deltas_db=eq_deltas,
        dynamic_unmask_delta_db=dynamic_unmask_delta_db,
    )
