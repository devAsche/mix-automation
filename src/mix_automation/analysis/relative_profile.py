"""Relative vocal-to-instrumental acoustic profiling and anchor propagation.

Instead of matching absolute EQ curves or static compressor thresholds (which break
immediately when backing track density changes), this module models invariant
relationships between the vocal and the backing track:
  1. Band-wise Vocal-to-Instrumental Margin (dB) across 6 critical mixing zones
  2. Crest-Factor Ratio & Short-Term Position Stability (vocal stickiness / glue)
  3. 2-5 kHz Resonance Prominence (harshness target for dynamic suppression)
  4. Section-to-Section Anchor Propagation (translating user tweaks from one section
     to louder/quieter sections automatically)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy import signal


VOCAL_MIX_BANDS: dict[str, tuple[float, float]] = {
    "sub_low": (40.0, 130.0),        # Rumble / unnecessary low collision
    "body_warmth": (160.0, 420.0),   # Core thickness / foundation against dense backing
    "boxiness_mud": (450.0, 950.0),  # Honk / muddiness control zone
    "core_presence": (1500.0, 4200.0),  # Forwardness, distance & vocal stickiness
    "sibilance": (5000.0, 8500.0),   # Japanese sa/ta & English s/sh/ch energy
    "air_sheen": (8500.0, 16000.0),  # Pro-level silky top-end extension
}

EPS = 1e-10


@dataclass(frozen=True)
class SectionRelativeProfile:
    """Song-invariant relative metrics between active vocal and backing track."""

    vocal_rms_db: float
    inst_rms_db: float
    broadband_vi_ratio_db: float
    vocal_crest_db: float
    inst_crest_db: float
    crest_difference_db: float
    position_stability_std_db: float
    harsh_resonance_index_db: float
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


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert (channels, samples) or (samples, channels) or 1D array to 1D mono."""
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False)
    if audio.shape[0] <= 2 and audio.shape[0] < audio.shape[1]:
        return np.mean(audio, axis=0, dtype=np.float64)
    return np.mean(audio, axis=1, dtype=np.float64)


def _db(power_or_amp: float | np.ndarray, is_power: bool = True) -> float | np.ndarray:
    scale = 10.0 if is_power else 20.0
    return scale * np.log10(np.maximum(power_or_amp, EPS))


def compute_section_relative_profile(
    vocal: np.ndarray,
    instrumental: np.ndarray,
    sr: int,
    frame_sec: float = 0.05,
    hop_sec: float = 0.025,
    activity_rel_threshold_db: float = -28.0,
) -> SectionRelativeProfile:
    """Extract the relative vocal-to-instrumental profile over active vocal frames."""
    v_mono = _to_mono(vocal)
    i_mono = _to_mono(instrumental)
    length = min(len(v_mono), len(i_mono))
    if length == 0:
        raise ValueError("Audio inputs must not be empty.")
    v_mono = v_mono[:length]
    i_mono = i_mono[:length]

    n_fft = int(2 ** np.ceil(np.log2(max(int(sr * frame_sec), 1024))))
    hop_length = max(int(sr * hop_sec), 256)

    freqs, _, v_stft = signal.stft(v_mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, i_stft = signal.stft(i_mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    v_pow = np.abs(v_stft) ** 2
    i_pow = np.abs(i_stft) ** 2

    # Identify active singing frames (ignore rests/interludes so instrumental solos don't skew ratios)
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

    # Frame-by-frame relative offset stability ("一の安定感" / position stability)
    frame_vi_diff_db = v_frame_db[active_mask] - _db(i_frame_power[active_mask], is_power=True)
    position_stability_std_db = float(np.std(frame_vi_diff_db))

    # Time-domain short-window crest factors over active frames
    v_peak_db = float(_db(np.percentile(np.abs(v_mono), 99.5), is_power=False))
    i_peak_db = float(_db(np.percentile(np.abs(i_mono), 99.5), is_power=False))
    v_time_rms_db = float(_db(np.sqrt(np.mean(v_mono**2) + EPS) / max(np.sqrt(active_ratio), 0.1), is_power=False))
    i_time_rms_db = float(_db(np.sqrt(np.mean(i_mono**2) + EPS), is_power=False))
    v_crest_db = max(v_peak_db - v_time_rms_db, 0.0)
    i_crest_db = max(i_peak_db - i_time_rms_db, 0.0)

    # Band-wise Vocal-to-Instrumental Margins (dB)
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

    # 2-5 kHz Narrow-band Resonance Prominence (Harshness / Soothe2 metric)
    harsh_mask = (freqs >= 2000.0) & (freqs <= 5000.0)
    v_harsh_spec_db = _db(np.mean(v_pow[harsh_mask][:, active_mask], axis=1), is_power=True)
    smooth_width = max(5, len(v_harsh_spec_db) // 8)
    kernel = np.ones(smooth_width) / smooth_width
    smoothed_env_db = np.convolve(v_harsh_spec_db, kernel, mode="same")
    prominence_db = np.maximum(v_harsh_spec_db - smoothed_env_db, 0.0)
    harsh_resonance_index_db = float(np.percentile(prominence_db, 90))

    return SectionRelativeProfile(
        vocal_rms_db=round(v_rms_active_db, 3),
        inst_rms_db=round(i_rms_active_db, 3),
        broadband_vi_ratio_db=round(broadband_vi_db, 3),
        vocal_crest_db=round(v_crest_db, 3),
        inst_crest_db=round(i_crest_db, 3),
        crest_difference_db=round(v_crest_db - i_crest_db, 3),
        position_stability_std_db=round(position_stability_std_db, 3),
        harsh_resonance_index_db=round(harsh_resonance_index_db, 3),
        band_margins_db=band_margins_db,
        inst_band_levels_db=inst_band_levels_db,
        active_frame_ratio=round(active_ratio, 3),
    )


def compute_anchor_propagation(
    anchor_profile: SectionRelativeProfile,
    target_raw_profile: SectionRelativeProfile,
    eq_clamp_db: float = 4.5,
) -> SectionTransferAdjustment:
    """Translate the user-approved mix balance from an anchor section to another section.

    When the backing track drops or explodes in loudness/density between Verse and Chorus,
    this computes the exact gain, compressor threshold, and tonal EQ deltas required so
    the target section achieves the exact same vocal-to-instrumental blend as the anchor.
    """
    # 1. Broadband gain offset required to match the anchor's V/I ratio
    gain_delta_db = anchor_profile.broadband_vi_ratio_db - target_raw_profile.broadband_vi_ratio_db

    # 2. Spectral tilt / band compensation after broadband gain shift
    eq_deltas: dict[str, float] = {}
    for band_name in VOCAL_MIX_BANDS:
        projected_margin = target_raw_profile.band_margins_db[band_name] + gain_delta_db
        needed_eq = anchor_profile.band_margins_db[band_name] - projected_margin
        # Dampen extreme boosts on sub_low and sibilance for safety
        weight = 0.4 if band_name in ("sub_low", "sibilance") else 0.75
        eq_deltas[band_name] = round(float(np.clip(needed_eq * weight, -eq_clamp_db, eq_clamp_db)), 2)

    # 3. Compressor threshold delta so denser sections get proportionally tighter glue
    crest_gap = target_raw_profile.crest_difference_db - anchor_profile.crest_difference_db
    comp_threshold_delta_db = round(float(np.clip(-0.65 * crest_gap, -6.0, 6.0)), 2)

    # 4. Instrumental dynamic unmasking depth delta for dense midranges
    presence_inst_jump = (
        target_raw_profile.inst_band_levels_db["core_presence"]
        - anchor_profile.inst_band_levels_db["core_presence"]
    )
    dynamic_unmask_delta_db = round(float(np.clip(0.35 * presence_inst_jump, -2.5, 3.0)), 2)

    return SectionTransferAdjustment(
        input_gain_delta_db=round(float(gain_delta_db), 2),
        compressor_threshold_delta_db=comp_threshold_delta_db,
        eq_band_deltas_db=eq_deltas,
        dynamic_unmask_delta_db=dynamic_unmask_delta_db,
    )
