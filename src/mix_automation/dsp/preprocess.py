"""Stage 1 Vocal Pre-processing: RX 10 cleanup, pitch stabilization, and phoneme-aware gain riding.

Implements the user's foundational vocal preparation workflow:
  1. Lip noise, mouth click, plosive, noise floor & breath control via real iZotope RX 10 VST3s
  2. Nuance-preserving pitch stabilization (leaves vibrato, scoops & falls intact while gently
     centering sustained vowels)
  3. Phoneme-aware & backing-track-adaptive pre-gain riding ("波形の太さを整えてオケに負けない太い土台を作る")
     so Japanese & English syllables enter the compressor at uniform, rock-solid density without
     boosting sibilants or breaths.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from pedalboard import Pedalboard, load_plugin
from scipy import ndimage, signal

from mix_automation.config.plugin_registry import get_plugin_map
from mix_automation.io.audio_loader import strip_silence_gate


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for RX 10 cleanup, subtle pitch centering, and pre-gain leveling."""

    enable_rx10_cleanup: bool = True
    mouth_declick_sensitivity: float = 4.2
    deplosive_sensitivity: float = 5.0
    deplosive_strength: float = 6.0
    breath_target_db: float = -18.0
    breath_sensitivity: float = 55.0
    enable_pitch_stabilizer: bool = True
    pitch_correction_strength: float = 0.40  # 0.0=off, 0.4=natural pro centering, 1.0=hard
    enable_gain_rider: bool = True
    rider_max_boost_db: float = 5.5
    rider_max_cut_db: float = -5.5
    rider_backing_follow: float = 0.45  # How strongly vocal foundation tracks backing loudness


def run_rx10_cleanup(mono_vocal: np.ndarray, sr: int, cfg: PreprocessConfig) -> np.ndarray:
    """Process dry vocal through licensed iZotope RX 10 VST3 restoration modules."""
    pmap = get_plugin_map()
    chain = []

    # 1. Plosive control
    if pmap["rx10_deplosive"].exists:
        deplosive = load_plugin(str(pmap["rx10_deplosive"].loadable_path))
        if "sensitivity" in deplosive.parameters:
            deplosive.sensitivity = cfg.deplosive_sensitivity
        if "strength" in deplosive.parameters:
            deplosive.strength = cfg.deplosive_strength
        chain.append(deplosive)

    # 2. Mouth De-click (lip & saliva clicks)
    if pmap["rx10_mouth_declick"].exists:
        declick = load_plugin(str(pmap["rx10_mouth_declick"].loadable_path))
        if "sensitivity" in declick.parameters:
            declick.sensitivity = cfg.mouth_declick_sensitivity
        chain.append(declick)

    # 3. Gentle Voice De-noise
    if pmap["rx10_voice_denoise"].exists:
        denoise = load_plugin(str(pmap["rx10_voice_denoise"].loadable_path))
        if "reduction" in denoise.parameters:
            denoise.reduction = 6.0
        chain.append(denoise)

    # 4. Breath Control (attenuate rather than mute to avoid unnatural choking)
    if pmap["rx10_breath_control"].exists:
        breath = load_plugin(str(pmap["rx10_breath_control"].loadable_path))
        if "gain_db" in breath.parameters:
            breath.gain_db = cfg.breath_target_db
        if "sensitivity" in breath.parameters:
            breath.sensitivity = cfg.breath_sensitivity
        chain.append(breath)

    if not chain:
        return mono_vocal

    board = Pedalboard(chain)
    # Process as 2ch stereo buffer if any RX module expects stereo I/O, then return mono
    stereo_in = np.stack([mono_vocal, mono_vocal], axis=0).astype(np.float32)
    stereo_out = board(stereo_in, sr)
    return np.mean(stereo_out, axis=0, dtype=np.float32)


def detect_key_scale_mask(inst_mono: np.ndarray, sr: int) -> np.ndarray:
    """Estimate the 12-pitch-class diatonic scale mask (boolean[12]) from the backing track."""
    chroma = librosa.feature.chroma_cqt(y=inst_mono, sr=sr, hop_length=1024)
    chroma_energy = np.mean(chroma, axis=1)

    # Major (Ionian) template; relative minor shares identical pitch classes
    base_major = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    best_score = -1.0
    best_mask = np.ones(12, dtype=bool)
    for root in range(12):
        rotated = np.roll(base_major, root)
        score = float(np.dot(chroma_energy, rotated))
        if score > best_score:
            best_score = score
            best_mask = rotated > 0.5
    return best_mask


def stabilize_vocal_pitch_nuance_safe(
    mono_vocal: np.ndarray,
    sr: int,
    scale_mask: np.ndarray | None = None,
    strength: float = 0.40,
) -> np.ndarray:
    """Center sustained vowel pitch gently while preserving scoops, falls, and vibrato.

    Operates on contiguous stable voiced segments only (80ms+ after note onset), avoiding
    phase-vocoder smearing on unvoiced consonants or fast melodic ornaments.
    """
    if strength <= 0.01:
        return mono_vocal

    hop = 512
    f0, voiced_flag, voiced_probs = librosa.pyin(
        mono_vocal,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        hop_length=hop,
    )
    midi = librosa.hz_to_midi(np.where(np.isnan(f0), 440.0, f0))
    valid = (~np.isnan(f0)) & voiced_flag & (voiced_probs > 0.72)

    out = mono_vocal.copy()
    labels, num_regions = ndimage.label(valid)
    min_frames = int(0.14 * sr / hop)  # Only touch sustained notes >= 140ms
    onset_protect = int(0.06 * sr / hop)
    release_protect = int(0.04 * sr / hop)

    for r_idx in range(1, num_regions + 1):
        idxs = np.where(labels == r_idx)[0]
        if len(idxs) < min_frames:
            continue

        core_idxs = idxs[onset_protect : len(idxs) - release_protect]
        if len(core_idxs) < 4:
            continue

        seg_midi = midi[core_idxs]
        # Skip intentional pitch glides / slides (std > 0.42 semitones)
        if np.std(seg_midi) > 0.42:
            continue

        median_midi = float(np.median(seg_midi))
        nearest_note = round(median_midi)
        if scale_mask is not None and not scale_mask[nearest_note % 12]:
            # If accidental/passing tone outside diatonic scale, check if very close anyway
            if abs(median_midi - nearest_note) > 0.30:
                continue

        cents_off = (median_midi - nearest_note) * 100.0
        # Only nudge notes that are slightly flat/sharp (between 10 and 45 cents)
        if not (10.0 <= abs(cents_off) <= 45.0):
            continue

        shift_semitones = -(median_midi - nearest_note) * strength
        s_start = max(0, core_idxs[0] * hop)
        s_end = min(len(mono_vocal), (core_idxs[-1] + 1) * hop)
        if s_end - s_start < 2048:
            continue

        chunk = mono_vocal[s_start:s_end]
        shifted = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=shift_semitones, n_fft=2048)

        # Crossfade edges seamlessly so zero clicks occur
        xf = min(512, len(chunk) // 4)
        fade_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)
        fade_out = 1.0 - fade_in
        blended = shifted.copy()
        blended[:xf] = chunk[:xf] * fade_out + shifted[:xf] * fade_in
        blended[-xf:] = shifted[-xf:] * fade_out + chunk[-xf:] * fade_in
        out[s_start:s_end] = blended

    return out.astype(np.float32)


def apply_phoneme_aware_gain_rider(
    mono_vocal: np.ndarray,
    inst_mono: np.ndarray,
    sr: int,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Level vocal body frame-by-frame relative to backing track while ignoring sibilants/breaths.

    Returns `(leveled_vocal, gain_db_curve)`.
    """
    n = min(len(mono_vocal), len(inst_mono))
    v = mono_vocal[:n]
    inst = inst_mono[:n]

    hop = int(sr * 0.02)  # 20ms resolution
    win = hop * 4         # 80ms window

    # Bandpass 160Hz - 3800Hz to measure pure voiced vowel core (ignores breaths/sibilants/rumble!)
    sos_vowel = signal.butter(4, [160.0, 3800.0], btype="bandpass", fs=sr, output="sos")
    v_vowel = signal.sosfiltfilt(sos_vowel, v)
    sos_high = signal.butter(4, 4800.0, btype="highpass", fs=sr, output="sos")
    v_sibilant = signal.sosfiltfilt(sos_high, v)

    # Frame envelopes
    n_frames = max(1, int(np.ceil(n / hop)))
    pad_len = n_frames * hop - n
    v_v_frames = np.pad(v_vowel, (0, pad_len)).reshape(n_frames, hop)
    v_s_frames = np.pad(v_sibilant, (0, pad_len)).reshape(n_frames, hop)
    i_frames = np.pad(inst, (0, pad_len)).reshape(n_frames, hop)

    v_core_rms = np.sqrt(np.mean(v_v_frames**2, axis=1) + 1e-12)
    v_sib_rms = np.sqrt(np.mean(v_s_frames**2, axis=1) + 1e-12)
    i_rms = np.sqrt(np.mean(i_frames**2, axis=1) + 1e-12)

    # Smooth instrumental envelope over ~1.5 seconds so vocal tracks section energy, not kick drum hits
    inst_smooth_frames = max(3, int(1.5 / 0.02))
    i_rms_smooth = ndimage.uniform_filter1d(i_rms, size=inst_smooth_frames)

    v_core_db = 20.0 * np.log10(v_core_rms)
    v_sib_db = 20.0 * np.log10(v_sib_rms)
    i_smooth_db = 20.0 * np.log10(i_rms_smooth)

    peak_v_db = float(np.percentile(v_core_db, 95))
    voiced_vowel_mask = (v_core_db > (peak_v_db - 28.0)) & ((v_core_db - v_sib_db) > -2.0)

    if not np.any(voiced_vowel_mask):
        return mono_vocal, np.zeros(n_frames, dtype=np.float32)

    median_v_db = float(np.median(v_core_db[voiced_vowel_mask]))
    median_i_db = float(np.median(i_smooth_db[voiced_vowel_mask]))

    # Target vocal vowel core tracks median + partial follow of backing track loudness jumps
    inst_variation_db = np.clip(i_smooth_db - median_i_db, -8.0, 8.0) * cfg.rider_backing_follow
    target_v_db = median_v_db + inst_variation_db

    raw_gain_db = np.zeros(n_frames, dtype=np.float32)
    desired = np.clip(
        (target_v_db - v_core_db) * 0.65,
        cfg.rider_max_cut_db,
        cfg.rider_max_boost_db,
    )
    raw_gain_db[voiced_vowel_mask] = desired[voiced_vowel_mask]

    # Hold gain across short consonants/rests instead of snapping abruptly to 0 dB
    last_g = 0.0
    for idx in range(n_frames):
        if voiced_vowel_mask[idx]:
            last_g = float(raw_gain_db[idx])
        else:
            last_g *= 0.92  # Gentle return toward unity during silence/breaths
            raw_gain_db[idx] = last_g

    # Smooth rider curve (~180ms window) for completely transparent, artifact-free leveling
    smooth_frames = max(3, int(0.18 / 0.02))
    smooth_gain_db = ndimage.gaussian_filter1d(raw_gain_db, sigma=smooth_frames / 2.5)

    sample_gain_db = np.interp(
        np.arange(len(mono_vocal)),
        np.arange(n_frames) * hop + hop // 2,
        smooth_gain_db,
    ).astype(np.float32)
    linear_gain = 10.0 ** (sample_gain_db / 20.0)

    return (mono_vocal * linear_gain).astype(np.float32), smooth_gain_db.astype(np.float32)


def preprocess_vocal_pipeline(
    raw_mono_vocal: np.ndarray,
    inst_mono: np.ndarray,
    sr: int,
    cfg: PreprocessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute full Stage 1 pre-processing and return `(prepared_vocal, rider_gain_db_curve)`."""
    cfg = cfg or PreprocessConfig()

    # 1. Strip silence noise between phrases
    v = strip_silence_gate(raw_mono_vocal, sr=sr)

    # 2. iZotope RX 10 Restoration Chain
    if cfg.enable_rx10_cleanup:
        v = run_rx10_cleanup(v, sr=sr, cfg=cfg)
        v = strip_silence_gate(v, sr=sr)

    # 3. Nuance-preserving Pitch Stabilization
    if cfg.enable_pitch_stabilizer:
        scale_mask = detect_key_scale_mask(inst_mono, sr=sr)
        v = stabilize_vocal_pitch_nuance_safe(
            v, sr=sr, scale_mask=scale_mask, strength=cfg.pitch_correction_strength
        )

    # 4. Phoneme-aware & Backing-adaptive Pre-Gain Riding ("太い土台作り")
    if cfg.enable_gain_rider:
        v, rider_db = apply_phoneme_aware_gain_rider(v, inst_mono, sr=sr, cfg=cfg)
    else:
        rider_db = np.zeros(1, dtype=np.float32)

    return v, rider_db
