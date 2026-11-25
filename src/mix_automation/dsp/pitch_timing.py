"""v5 Non-Destructive Pitch & Timing Intelligence: Melodyne Protection, Smooth Praat PitchTier & Safe Markers.

Fixes v4 timing/pitch glitches structurally:
  1. Never alters timing on Melodyne-edited or human-grooved tracks unless `enable_auto_align=True`
     is explicitly requested! Previously, slipping `mainvocal` alone by +-85ms against a Vocaloid
     original caused rushed/crammed phrasing and desynced the lead from recorded harmonies.
  2. Emits high-confidence Studio One 5 `<MarkerEvent>` cues for genuine timing/pitch outliers so
     the user retains 100% creative groove control.
  3. For untuned vocals (or `--auto-pitch`), performs continuous Praat (`praat-parselmouth`)
     PitchTier overlap-add resynthesis — preserving unvoiced consonants, scoops, and vibrato intact
     while gently centering sustained steady pitch drift (never hard-splicing audio chunks!).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ReviewMarker:
    """Timeline marker injected into Studio One 5's MarkerTrack for instant human review."""

    time_sec: float
    name: str
    category: str  # "pitch" | "timing" | "fx"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PitchTimingReport:
    """Summary of automatic pitch/timing inspection and conditional processing."""

    median_f0_hz: float
    initial_median_cents_off: float
    already_melodyne_tuned: bool
    pitch_applied: bool
    phrases_slipped_count: int
    notes_nudged_count: int
    review_markers: list[ReviewMarker]


def analyze_vocal_f0_and_precision(mono_vocal: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return `(f0_hz, voiced_mask, median_f0_hz, median_abs_cents_off)`."""
    hop = 512
    f0, voiced_flag, voiced_prob = librosa.pyin(
        mono_vocal,
        fmin=70.0,
        fmax=900.0,
        sr=sr,
        hop_length=hop,
    )
    valid = (~np.isnan(f0)) & voiced_flag & (voiced_prob > 0.75)
    if not np.any(valid):
        return f0, valid, 240.0, 0.0

    med_f0 = float(np.median(f0[valid]))
    midi = librosa.hz_to_midi(f0[valid])
    cents_off = np.abs((midi - np.round(midi)) * 100.0)
    med_cents = float(np.median(cents_off))
    return f0, valid, round(med_f0, 1), round(med_cents, 2)


def align_phrase_onsets_zero_stretch(
    vocal_mono: np.ndarray,
    sr: int,
    ref_vocal_mono: np.ndarray | None,
    apply_audio_shift: bool = False,
    max_slip_ms: float = 36.0,
    min_trigger_ms: float = 22.0,
) -> tuple[np.ndarray, int, list[ReviewMarker]]:
    """Inspect phrase onsets vs reference vocal; emit review markers and ONLY slip audio when explicitly enabled."""
    markers: list[ReviewMarker] = []
    if ref_vocal_mono is None or len(ref_vocal_mono) < sr * 5:
        return vocal_mono, 0, markers

    n = min(len(vocal_mono), len(ref_vocal_mono))
    hop = int(sr * 0.01)  # 10ms envelope resolution
    n_f = n // hop
    if n_f < 50:
        return vocal_mono, 0, markers

    v_env = np.sqrt(np.mean(vocal_mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1) + 1e-12)
    r_env = np.sqrt(np.mean(ref_vocal_mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1) + 1e-12)

    v_act = v_env > (np.percentile(v_env, 96) * 0.04)
    r_act = r_env > (np.percentile(r_env, 96) * 0.05)

    # Dilate slightly (100ms) to group syllables into whole phrases
    phrase_mask = ndimage.binary_closing(v_act, structure=np.ones(10))
    labels, n_phrases = ndimage.label(phrase_mask)

    out = np.zeros_like(vocal_mono) if apply_audio_shift else vocal_mono.copy()
    if apply_audio_shift:
        out[n:] = vocal_mono[n:]

    slipped_count = 0
    max_lag_frames = int(max_slip_ms / 10.0)
    min_lag_frames = int(min_trigger_ms / 10.0)
    marker_scan_frames = 10  # Scan +-100ms for diagnostic review markers

    for p_i in range(1, n_phrases + 1):
        idxs = np.where(labels == p_i)[0]
        f0_idx, f1_idx = int(idxs[0]), int(idxs[-1])
        s0 = f0_idx * hop
        s1 = min(n, (f1_idx + 1) * hop)

        left_gap = f0_idx - (np.where(phrase_mask[:f0_idx])[0][-1] if np.any(phrase_mask[:f0_idx]) else 0)
        right_hits = np.where(phrase_mask[f1_idx + 1 :])[0]
        right_gap = int(right_hits[0]) if len(right_hits) > 0 else 50

        if (f1_idx - f0_idx) < 35 or not np.any(r_act[f0_idx:f1_idx]):
            if apply_audio_shift:
                out[s0:s1] += vocal_mono[s0:s1]
            continue

        v_diff = np.maximum(np.diff(v_env[f0_idx:f1_idx], prepend=0.0), 0.0)
        r_diff = np.maximum(np.diff(r_env[f0_idx:f1_idx], prepend=0.0), 0.0)
        v_norm = float(np.linalg.norm(v_diff)) + 1e-9
        r_norm = float(np.linalg.norm(r_diff)) + 1e-9

        best_shift_f = 0
        best_corr = -1.0
        for lag in range(-marker_scan_frames, marker_scan_frames + 1):
            if lag < 0:
                c = float(np.dot(v_diff[-lag:], r_diff[: len(v_diff) + lag])) / (v_norm * r_norm)
            elif lag > 0:
                c = float(np.dot(v_diff[:-lag], r_diff[lag:])) / (v_norm * r_norm)
            else:
                c = float(np.dot(v_diff, r_diff)) / (v_norm * r_norm)
            if c > best_corr:
                best_corr = c
                best_shift_f = lag

        if best_corr >= 0.55 and abs(best_shift_f) >= 9:  # >= 90ms genuine discrepancy -> review marker only
            markers.append(
                ReviewMarker(
                    time_sec=round(f0_idx * 0.01, 2),
                    name=f"タイミング確認 ({-best_shift_f * 10:+d}ms)",
                    category="timing",
                )
            )

        # ONLY shift audio if explicitly requested AND generous 280ms silence margins exist AND high correlation!
        if (
            apply_audio_shift
            and best_corr >= 0.62
            and left_gap >= 28
            and right_gap >= 28
            and min_lag_frames <= abs(best_shift_f) <= max_lag_frames
        ):
            safe_shift_f = int(np.clip(best_shift_f, -max_lag_frames, max_lag_frames))
            shift_samples = safe_shift_f * hop
            dst0 = max(0, s0 + shift_samples)
            dst1 = min(n, s1 + shift_samples)
            copy_len = min(s1 - s0, dst1 - dst0)
            seg = vocal_mono[s0 : s0 + copy_len].copy()
            fade = min(int(0.025 * sr), copy_len // 4)
            if fade > 4:
                seg[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            out[dst0 : dst0 + copy_len] += seg
            slipped_count += 1
        elif apply_audio_shift:
            out[s0:s1] += vocal_mono[s0:s1]

    return out.astype(np.float32), slipped_count, markers


def apply_td_psola_nuance_pitch_correction(
    vocal_mono: np.ndarray,
    sr: int,
    f0: np.ndarray,
    valid_mask: np.ndarray,
    scale_mask: np.ndarray,
    strength: float = 0.50,
) -> tuple[np.ndarray, int, list[ReviewMarker]]:
    """Smooth continuous Praat PitchTier correction preserving scoops, vibrato & consonants."""
    import parselmouth
    from parselmouth.praat import call

    hop = 512
    midi = librosa.hz_to_midi(np.where(np.isnan(f0), 220.0, f0))
    labels, n_reg = ndimage.label(valid_mask)

    min_frames = int(0.18 * sr / hop)
    onset_protect = int(0.075 * sr / hop)
    release_protect = int(0.055 * sr / hop)

    # Build smooth frame-level shift multiplier curve (1.0 everywhere except sustained slightly-off vowels!)
    shift_ratio_frames = np.ones(len(f0), dtype=np.float64)
    nudged = 0
    markers: list[ReviewMarker] = []

    for r_i in range(1, n_reg + 1):
        idxs = np.where(labels == r_i)[0]
        if len(idxs) < min_frames:
            continue

        core = idxs[onset_protect : len(idxs) - release_protect]
        if len(core) < 5:
            continue

        seg_midi = midi[core]
        if np.std(seg_midi) > 0.34:
            continue  # Intentional wide vibrato or melodic slide -> preserve 100% untouched!

        med_m = float(np.median(seg_midi))
        nearest = round(med_m)
        cents_off = (med_m - nearest) * 100.0

        if abs(cents_off) >= 42.0:
            markers.append(
                ReviewMarker(
                    time_sec=round(float(idxs[0] * hop / sr), 2),
                    name=f"ピッチ要確認 ({cents_off:+.0f}ct)",
                    category="pitch",
                )
            )
            continue

        if not scale_mask[nearest % 12] or abs(cents_off) < 14.0:
            continue

        shift_st = float(np.clip(-(med_m - nearest) * strength, -0.35, 0.35))
        ramp = np.ones(len(core), dtype=np.float64)
        edge = min(4, len(core) // 3)
        if edge > 1:
            ramp[:edge] = np.linspace(0.0, 1.0, edge)
            ramp[-edge:] = np.linspace(1.0, 0.0, edge)
        shift_ratio_frames[core] = 2.0 ** ((shift_st * ramp) / 12.0)
        nudged += 1

    if nudged == 0:
        return vocal_mono.astype(np.float32), 0, markers

    try:
        snd = parselmouth.Sound(vocal_mono.astype(np.float64), sampling_frequency=float(sr))
        manip = call(snd, "To Manipulation", 0.01, 75.0, 750.0)
        pitch_tier = call(manip, "Extract pitch tier")
        n_pts = int(call(pitch_tier, "Get number of points"))
        for idx_p in range(1, n_pts + 1):
            t_p = float(call(pitch_tier, "Get time from index", idx_p))
            f_p = float(call(pitch_tier, "Get value at index", idx_p))
            f_idx = int(np.clip(round(t_p * sr / hop), 0, len(shift_ratio_frames) - 1))
            r = float(shift_ratio_frames[f_idx])
            if abs(r - 1.0) > 1e-4:
                call(pitch_tier, "Remove point", idx_p)
                call(pitch_tier, "Add point", t_p, f_p * r)
        call([manip, pitch_tier], "Replace pitch tier")
        resynth = call(manip, "Get resynthesis (overlap-add)")
        y = np.asarray(resynth.values[0], dtype=np.float32)
        if len(y) >= len(vocal_mono):
            y = y[: len(vocal_mono)]
        else:
            y = np.pad(y, (0, len(vocal_mono) - len(y)))
        if np.all(np.isfinite(y)):
            return y, nudged, markers
    except Exception:
        pass

    return vocal_mono.astype(np.float32), 0, markers


def run_smart_pitch_and_timing_stage(
    vocal_mono: np.ndarray,
    sr: int,
    scale_mask: np.ndarray,
    ref_vocal_mono: np.ndarray | None = None,
    force_pitch: bool = False,
    enable_auto_align: bool = False,
) -> tuple[np.ndarray, PitchTimingReport]:
    """Inspect vocal precision; preserve Melodyne/human groove by default and emit Studio One markers."""
    # 1. Inspect vocal register & existing Melodyne tuning precision FIRST!
    f0, valid_mask, med_f0, med_cents = analyze_vocal_f0_and_precision(vocal_mono, sr=sr)
    already_tuned = (med_cents <= 20.0) and not force_pitch

    # 2. Timing: NEVER slip Melodyne-edited takes; only shift audio when explicitly enabled on untuned vocals!
    aligned_vocal, slipped_cnt, timing_markers = align_phrase_onsets_zero_stretch(
        vocal_mono,
        sr=sr,
        ref_vocal_mono=ref_vocal_mono,
        apply_audio_shift=(enable_auto_align and not already_tuned),
    )

    pitch_markers: list[ReviewMarker] = []
    nudged_cnt = 0

    if already_tuned:
        midi = librosa.hz_to_midi(np.where(np.isnan(f0), 220.0, f0))
        labels, n_reg = ndimage.label(valid_mask)
        min_f = int(0.18 * sr / 512)
        for r_i in range(1, n_reg + 1):
            idxs = np.where(labels == r_i)[0]
            if len(idxs) >= min_f:
                seg = midi[idxs[3:-2]] if len(idxs) > 6 else midi[idxs]
                if len(seg) > 0 and np.std(seg) < 0.35:
                    c_off = (float(np.median(seg)) - round(float(np.median(seg)))) * 100.0
                    if abs(c_off) >= 45.0:
                        pitch_markers.append(
                            ReviewMarker(
                                time_sec=round(float(idxs[0] * 512 / sr), 2),
                                name=f"ピッチ確認 ({c_off:+.0f}ct)",
                                category="pitch",
                            )
                        )
        final_vocal = aligned_vocal
    else:
        if slipped_cnt > 0:
            f0, valid_mask, med_f0, med_cents = analyze_vocal_f0_and_precision(aligned_vocal, sr=sr)
        final_vocal, nudged_cnt, pitch_markers = apply_td_psola_nuance_pitch_correction(
            aligned_vocal, sr=sr, f0=f0, valid_mask=valid_mask, scale_mask=scale_mask
        )

    all_markers = sorted(timing_markers + pitch_markers, key=lambda m: m.time_sec)[:24]
    report = PitchTimingReport(
        median_f0_hz=med_f0,
        initial_median_cents_off=med_cents,
        already_melodyne_tuned=already_tuned,
        pitch_applied=(not already_tuned and nudged_cnt > 0),
        phrases_slipped_count=slipped_cnt,
        notes_nudged_count=nudged_cnt,
        review_markers=all_markers,
    )
    return final_vocal, report
