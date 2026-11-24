"""v2 Stage 1 Vocal Pre-processing: RX 10 Restoration, Smart Slip/PSOLA & Standard -18 LUFS Staging.

Strictly adheres to the "Do No Harm" audio engineering principle:
  - Zero phase-vocoder (`librosa`) pitch shifting or crude custom STFT masks.
  - Genuine iZotope RX 10 VST3 chain (`De-plosive` -> `Mouth De-click` -> `Voice De-noise` -> `Breath Control`).
  - Reference-guided zero-stretch phrase slip alignment & conditional Praat TD-PSOLA micro-tuning
    (automatically bypassed when the stem is already Melodyne-tuned!).
  - Phoneme-aware vowel-core pre-gain riding + genuine `RX 10 De-ess` & gentle `Oeksound Soothe2`.
  - Standardized `-18.0 LUFS` active vocal gain staging so downstream VST3 compressors in both
    offline rendering and Studio One 5 hit the exact same sweet spot with ~0 dB channel faders!
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import pyloudnorm as pyln
from pedalboard import Pedalboard, load_plugin
from scipy import ndimage, signal

from mix_automation.config.plugin_registry import get_plugin_map
from mix_automation.dsp.pitch_timing import PitchTimingReport, run_smart_pitch_and_timing_stage
from mix_automation.io.audio_loader import strip_silence_gate


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for v2 clean pre-processing & gain staging."""

    enable_rx10_cleanup: bool = True
    mouth_declick_sensitivity: float = 4.0
    deplosive_sensitivity: float = 4.8
    deplosive_strength: float = 5.5
    breath_target_db: float = -18.5
    breath_sensitivity: float = 52.0
    force_auto_pitch: bool = False  # Auto-detects if already Melodyne-tuned unless forced
    enable_gain_rider: bool = True
    rider_max_boost_db: float = 4.5
    rider_max_cut_db: float = -4.5
    rider_backing_follow: float = 0.40
    deess_threshold_db: float = -21.5
    soothe_depth: float = 1.35      # Gentle genuine Soothe2 resonance taming (scale 0..5+)
    target_staged_lufs: float = -18.0


def _safe_set_vst(plug, key: str, val) -> None:
    if plug is None or key not in plug.parameters:
        return
    try:
        vv = getattr(plug.parameters[key], "valid_values", None)
        if vv and not isinstance(val, (bool, str)) and isinstance(vv[0], (int, float)) and not isinstance(vv[0], bool):
            val = min(vv, key=lambda x: abs(float(x) - float(val)))
        setattr(plug, key, val)
    except Exception:
        pass


def run_rx10_cleanup(mono_vocal: np.ndarray, sr: int, cfg: PreprocessConfig) -> np.ndarray:
    """Process dry vocal through licensed iZotope RX 10 VST3 restoration modules."""
    pmap = get_plugin_map()
    chain = []

    if pmap["rx10_deplosive"].exists:
        deplosive = load_plugin(str(pmap["rx10_deplosive"].loadable_path))
        _safe_set_vst(deplosive, "sensitivity", cfg.deplosive_sensitivity)
        _safe_set_vst(deplosive, "strength", cfg.deplosive_strength)
        chain.append(deplosive)

    if pmap["rx10_mouth_declick"].exists:
        declick = load_plugin(str(pmap["rx10_mouth_declick"].loadable_path))
        _safe_set_vst(declick, "sensitivity", cfg.mouth_declick_sensitivity)
        chain.append(declick)

    if pmap["rx10_voice_denoise"].exists:
        denoise = load_plugin(str(pmap["rx10_voice_denoise"].loadable_path))
        _safe_set_vst(denoise, "reduction", 5.0)
        chain.append(denoise)

    if pmap["rx10_breath_control"].exists:
        breath = load_plugin(str(pmap["rx10_breath_control"].loadable_path))
        _safe_set_vst(breath, "gain_db", cfg.breath_target_db)
        _safe_set_vst(breath, "sensitivity", cfg.breath_sensitivity)
        chain.append(breath)

    if not chain:
        return mono_vocal

    board = Pedalboard(chain)
    stereo_in = np.stack([mono_vocal, mono_vocal], axis=0).astype(np.float32)
    stereo_out = board(stereo_in, sr)
    return np.mean(stereo_out, axis=0, dtype=np.float32)


def detect_key_scale_mask(inst_mono: np.ndarray, sr: int) -> tuple[np.ndarray, int, str]:
    """Estimate the 12-pitch-class diatonic scale mask, root pitch class (0=C..11=B), and mode."""
    chroma = librosa.feature.chroma_cqt(y=inst_mono[: min(len(inst_mono), sr * 90)], sr=sr, hop_length=1024)
    chroma_energy = np.mean(chroma, axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_mask = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    minor_mask = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=bool)

    best_score = -1e9
    best_mask = major_mask
    best_root = 0
    best_mode = "Major"

    for root in range(12):
        sc_maj = float(np.corrcoef(chroma_energy, np.roll(major_profile, root))[0, 1])
        if sc_maj > best_score:
            best_score = sc_maj
            best_mask = np.roll(major_mask, root)
            best_root = root
            best_mode = "Major"

        sc_min = float(np.corrcoef(chroma_energy, np.roll(minor_profile, root))[0, 1])
        if sc_min > best_score:
            best_score = sc_min
            best_mask = np.roll(minor_mask, root)
            best_root = root
            best_mode = "Minor"

    return best_mask, best_root, best_mode


def apply_phoneme_aware_gain_rider(
    mono_vocal: np.ndarray,
    inst_mono: np.ndarray,
    sr: int,
    cfg: PreprocessConfig,
) -> np.ndarray:
    """Smoothly level voiced vowel core relative to backing track loudness without lifting sibilants."""
    n = min(len(mono_vocal), len(inst_mono))
    v = mono_vocal[:n]
    inst = inst_mono[:n]

    hop = int(sr * 0.02)
    sos_vowel = signal.butter(4, [160.0, 3600.0], btype="bandpass", fs=sr, output="sos")
    v_vowel = signal.sosfiltfilt(sos_vowel, v)
    sos_high = signal.butter(4, 4800.0, btype="highpass", fs=sr, output="sos")
    v_sibilant = signal.sosfiltfilt(sos_high, v)

    n_frames = max(1, int(np.ceil(n / hop)))
    pad_len = n_frames * hop - n
    v_v_frames = np.pad(v_vowel, (0, pad_len)).reshape(n_frames, hop)
    v_s_frames = np.pad(v_sibilant, (0, pad_len)).reshape(n_frames, hop)
    i_frames = np.pad(inst, (0, pad_len)).reshape(n_frames, hop)

    v_core_rms = np.sqrt(np.mean(v_v_frames**2, axis=1) + 1e-12)
    v_sib_rms = np.sqrt(np.mean(v_s_frames**2, axis=1) + 1e-12)
    i_rms = np.sqrt(np.mean(i_frames**2, axis=1) + 1e-12)

    inst_smooth_frames = max(3, int(1.6 / 0.02))
    i_rms_smooth = ndimage.uniform_filter1d(i_rms, size=inst_smooth_frames)

    v_core_db = 20.0 * np.log10(v_core_rms)
    v_sib_db = 20.0 * np.log10(v_sib_rms)
    i_smooth_db = 20.0 * np.log10(i_rms_smooth)

    peak_v_db = float(np.percentile(v_core_db, 95))
    voiced_vowel_mask = (v_core_db > (peak_v_db - 26.0)) & ((v_core_db - v_sib_db) > -1.5)
    if not np.any(voiced_vowel_mask):
        return mono_vocal

    median_v_db = float(np.median(v_core_db[voiced_vowel_mask]))
    median_i_db = float(np.median(i_smooth_db[voiced_vowel_mask]))

    inst_variation_db = np.clip(i_smooth_db - median_i_db, -7.0, 7.0) * cfg.rider_backing_follow
    target_v_db = median_v_db + inst_variation_db

    raw_gain_db = np.zeros(n_frames, dtype=np.float32)
    desired = np.clip(
        (target_v_db - v_core_db) * 0.58,
        cfg.rider_max_cut_db,
        cfg.rider_max_boost_db,
    )
    raw_gain_db[voiced_vowel_mask] = desired[voiced_vowel_mask]

    last_g = 0.0
    for idx in range(n_frames):
        if voiced_vowel_mask[idx]:
            last_g = float(raw_gain_db[idx])
        else:
            last_g *= 0.93
            raw_gain_db[idx] = last_g

    smooth_frames = max(4, int(0.22 / 0.02))
    smooth_gain_db = ndimage.gaussian_filter1d(raw_gain_db, sigma=smooth_frames / 2.5)
    sample_gain_db = np.interp(
        np.arange(len(mono_vocal)),
        np.arange(n_frames) * hop + hop // 2,
        smooth_gain_db,
    ).astype(np.float32)

    return (mono_vocal * (10.0 ** (sample_gain_db / 20.0))).astype(np.float32)


def run_genuine_deess_and_soothe2(mono_vocal: np.ndarray, sr: int, cfg: PreprocessConfig) -> np.ndarray:
    """Apply genuine commercial RX 10 De-ess and Oeksound Soothe2 VST3 plugins."""
    pmap = get_plugin_map()
    chain = []

    if pmap["rx10_deess"].exists:
        deess = load_plugin(str(pmap["rx10_deess"].loadable_path))
        _safe_set_vst(deess, "threshold", cfg.deess_threshold_db)
        chain.append(deess)

    if pmap["soothe2"].exists:
        try:
            s2 = load_plugin(str(pmap["soothe2"].loadable_path))
            _safe_set_vst(s2, "depth", cfg.soothe_depth)
            chain.append(s2)
        except Exception:
            pass

    if not chain:
        return mono_vocal

    board = Pedalboard(chain)
    st_in = np.stack([mono_vocal, mono_vocal], axis=0).astype(np.float32)
    st_out = board(st_in, sr)
    return np.mean(st_out, axis=0, dtype=np.float32)


def stage_vocal_loudness(mono_vocal: np.ndarray, sr: int, target_lufs: float = -18.0) -> np.ndarray:
    """Normalize active vocal loudness to exact `-18.0 LUFS` standard for consistent VST3 compressor drive."""
    meter = pyln.Meter(sr)
    # Measure integrated loudness (ITU-R BS.1770-4 automatically gates silent blocks!)
    cur_lufs = meter.integrated_loudness(mono_vocal.astype(np.float64))
    if not np.isfinite(cur_lufs):
        return mono_vocal
    gain_db = float(np.clip(target_lufs - cur_lufs, -18.0, 26.0))
    staged = mono_vocal * (10.0 ** (gain_db / 20.0))

    # Ensure clean headroom ceiling (-1.5 dBFS) without hard clipping
    peak = float(np.max(np.abs(staged)))
    max_lin = 10.0 ** (-1.5 / 20.0)
    if peak > max_lin:
        staged *= max_lin / peak
    return staged.astype(np.float32)


def preprocess_vocal_pipeline(
    raw_mono_vocal: np.ndarray,
    inst_mono: np.ndarray,
    sr: int,
    ref_vocal_mono: np.ndarray | None = None,
    cfg: PreprocessConfig | None = None,
) -> tuple[np.ndarray, PitchTimingReport, tuple[np.ndarray, int, str]]:
    """Execute complete Stage 1 clean pre-processing + gain staging."""
    cfg = cfg or PreprocessConfig()

    # 1. Gentle phrase noise gate
    v = strip_silence_gate(raw_mono_vocal, sr=sr)

    # 2. Genuine iZotope RX 10 cleanup
    if cfg.enable_rx10_cleanup:
        v = run_rx10_cleanup(v, sr=sr, cfg=cfg)
        v = strip_silence_gate(v, sr=sr)

    # 3. Key/Scale detection + Smart Slip-Timing & Conditional TD-PSOLA Micro-Pitch
    scale_info = detect_key_scale_mask(inst_mono, sr=sr)
    scale_mask, _, _ = scale_info
    v, pt_report = run_smart_pitch_and_timing_stage(
        v,
        sr=sr,
        scale_mask=scale_mask,
        ref_vocal_mono=ref_vocal_mono,
        force_pitch=cfg.force_auto_pitch,
    )

    # 4. Phoneme-aware Vowel-Core Gain Riding
    if cfg.enable_gain_rider:
        v = apply_phoneme_aware_gain_rider(v, inst_mono, sr=sr, cfg=cfg)

    # 5. Genuine RX 10 De-ess + Oeksound Soothe2 resonance polish
    v = run_genuine_deess_and_soothe2(v, sr=sr, cfg=cfg)

    # 6. Standard -18.0 LUFS Gain Staging
    v = stage_vocal_loudness(v, sr=sr, target_lufs=cfg.target_staged_lufs)

    return v, pt_report, scale_info
