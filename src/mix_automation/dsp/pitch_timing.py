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
from typing import Any

import librosa
import numpy as np
from scipy import ndimage

from mix_automation.dsp.lyrics_aligner import align_lyrics_and_phonemes, find_lyric_snippet_at_span


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


def _format_marker_name(base: str, lyrics_alignment: Any | None, time_sec: float, span_sec: float = 0.6) -> str:
    """Attach a short aligned lyric snippet to a Studio One review marker when available."""
    if lyrics_alignment is None:
        return base
    snippet = find_lyric_snippet_at_span(lyrics_alignment, time_sec, time_sec + span_sec)
    return f"{base} 「{snippet}」" if snippet else base


def _map_consonant_protect_to_frames(
    lyrics_alignment: Any | None,
    n_frames: int,
    hop: int,
    sr: int,
) -> np.ndarray:
    """Map alignment's `consonant_attack_protect_mask` onto an arbitrary `hop`-sample frame grid."""
    out = np.zeros(n_frames, dtype=bool)
    if lyrics_alignment is None:
        return out
    raw_mask = getattr(lyrics_alignment, "consonant_attack_protect_mask", None)
    if raw_mask is None or len(raw_mask) == 0:
        return out
    mask_arr = np.asarray(raw_mask, dtype=bool)
    align_hop = max(1, int(getattr(lyrics_alignment, "hop_samples", hop)))
    align_sr = max(1, int(getattr(lyrics_alignment, "sr", sr)))
    for idx in range(n_frames):
        t_sec = (idx * hop) / float(sr)
        a_idx = int(np.clip(round(t_sec * align_sr / align_hop), 0, len(mask_arr) - 1))
        out[idx] = bool(mask_arr[a_idx])
    return out


def _resynth_pitch_tier_praat(
    vocal_mono: np.ndarray,
    sr: int,
    hop: int,
    shift_ratio_frames: np.ndarray,
    lyrics_alignment: Any | None = None,
) -> tuple[np.ndarray, bool]:
    """Apply continuous Praat PitchTier scaling while strictly clamping protected consonants to 1.0."""
    import parselmouth
    from parselmouth.praat import call

    protect_mask = None
    align_hop = hop
    align_sr = sr
    if lyrics_alignment is not None:
        raw_m = getattr(lyrics_alignment, "consonant_attack_protect_mask", None)
        if raw_m is not None and len(raw_m) > 0:
            protect_mask = np.asarray(raw_m, dtype=bool)
            align_hop = max(1, int(getattr(lyrics_alignment, "hop_samples", hop)))
            align_sr = max(1, int(getattr(lyrics_alignment, "sr", sr)))

    try:
        snd = parselmouth.Sound(vocal_mono.astype(np.float64), sampling_frequency=float(sr))
        manip = call(snd, "To Manipulation", 0.01, 75.0, 750.0)
        pitch_tier = call(manip, "Extract pitch tier")
        n_pts = int(call(pitch_tier, "Get number of points"))
        if n_pts <= 0:
            return vocal_mono.astype(np.float32), False

        pts = [
            (
                float(call(pitch_tier, "Get time from index", idx_p)),
                float(call(pitch_tier, "Get value at index", idx_p)),
            )
            for idx_p in range(1, n_pts + 1)
        ]
        duration_sec = len(vocal_mono) / float(sr)
        call(pitch_tier, "Remove points between", 0.0, duration_sec + 1.0)

        for t_p, f_p in pts:
            f_idx = int(np.clip(round(t_p * sr / hop), 0, len(shift_ratio_frames) - 1))
            r = float(shift_ratio_frames[f_idx])
            if protect_mask is not None:
                a_idx = int(np.clip(round(t_p * align_sr / align_hop), 0, len(protect_mask) - 1))
                if protect_mask[a_idx]:
                    r = 1.0
            call(pitch_tier, "Add point", t_p, f_p * r)

        call([manip, pitch_tier], "Replace pitch tier")
        resynth = call(manip, "Get resynthesis (overlap-add)")
        y = np.asarray(resynth.values[0], dtype=np.float32)
        if len(y) >= len(vocal_mono):
            y = y[: len(vocal_mono)]
        else:
            y = np.pad(y, (0, len(vocal_mono) - len(y)))
        if np.all(np.isfinite(y)):
            return y, True
    except Exception:
        pass
    return vocal_mono.astype(np.float32), False


def align_phrase_onsets_zero_stretch(
    vocal_mono: np.ndarray,
    sr: int,
    ref_vocal_mono: np.ndarray | None,
    apply_audio_shift: bool = False,
    max_slip_ms: float = 36.0,
    min_trigger_ms: float = 22.0,
    lyrics_alignment: Any | None = None,
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
            t_sec = round(f0_idx * 0.01, 2)
            markers.append(
                ReviewMarker(
                    time_sec=t_sec,
                    name=_format_marker_name(f"タイミング確認 ({-best_shift_f * 10:+d}ms)", lyrics_alignment, t_sec),
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
    lyrics_alignment: Any | None = None,
) -> tuple[np.ndarray, int, list[ReviewMarker]]:
    """Smooth continuous Praat PitchTier correction preserving scoops, vibrato & consonants."""
    hop = 512
    midi = librosa.hz_to_midi(np.where(np.isnan(f0), 220.0, f0))
    frame_protect = _map_consonant_protect_to_frames(lyrics_alignment, len(f0), hop=hop, sr=sr)
    labels, n_reg = ndimage.label(valid_mask)

    min_frames = int(0.18 * sr / hop)
    onset_protect = int(0.045 * sr / hop)
    release_protect = int(0.035 * sr / hop)

    # Build smooth frame-level shift multiplier curve (1.0 everywhere except sustained slightly-off vowels!)
    shift_ratio_frames = np.ones(len(f0), dtype=np.float64)
    nudged = 0
    markers: list[ReviewMarker] = []

    for r_i in range(1, n_reg + 1):
        idxs = np.where(labels == r_i)[0]
        if len(idxs) < min_frames:
            continue

        core = idxs[onset_protect : len(idxs) - release_protect]
        if np.any(frame_protect):
            unprot = core[~frame_protect[core]]
            if len(unprot) >= 5:
                core = unprot
            else:
                continue
        if len(core) < 5:
            continue

        seg_midi = midi[core]
        if np.std(seg_midi) > 0.34:
            continue  # Intentional wide vibrato or melodic slide -> preserve 100% untouched!

        med_m = float(np.median(seg_midi))
        nearest = round(med_m)
        cents_off = (med_m - nearest) * 100.0

        if abs(cents_off) >= 45.0:
            t_sec = round(float(idxs[0] * hop / sr), 2)
            markers.append(
                ReviewMarker(
                    time_sec=t_sec,
                    name=_format_marker_name(f"ピッチ確認 ({cents_off:+.0f}ct)", lyrics_alignment, t_sec),
                    category="pitch",
                )
            )
            continue

        if not scale_mask[nearest % 12] or abs(cents_off) < 14.0:
            continue

        shift_st = float(np.clip(-(med_m - nearest) * strength, -0.42, 0.42))
        ramp = np.ones(len(core), dtype=np.float64)
        edge = min(4, len(core) // 3)
        if edge > 1:
            ramp[:edge] = np.linspace(0.0, 1.0, edge)
            ramp[-edge:] = np.linspace(1.0, 0.0, edge)
        shift_ratio_frames[core] = 2.0 ** ((shift_st * ramp) / 12.0)
        nudged += 1

    # Strictly clamp ALL consonant-protected frames to ratio 1.0 (zero resynthesis smear!)
    shift_ratio_frames[frame_protect] = 1.0

    if nudged == 0:
        return vocal_mono.astype(np.float32), 0, markers

    y, ok = _resynth_pitch_tier_praat(
        vocal_mono, sr=sr, hop=hop, shift_ratio_frames=shift_ratio_frames, lyrics_alignment=lyrics_alignment
    )
    if ok:
        return y, nudged, markers
    return vocal_mono.astype(np.float32), 0, markers


def run_smart_pitch_and_timing_stage(
    vocal_mono: np.ndarray,
    sr: int,
    scale_mask: np.ndarray,
    ref_vocal_mono: np.ndarray | None = None,
    force_pitch: bool = False,
    enable_auto_align: bool = False,
    lyrics_alignment: Any | None = None,
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
        lyrics_alignment=lyrics_alignment,
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
                        t_sec = round(float(idxs[0] * 512 / sr), 2)
                        pitch_markers.append(
                            ReviewMarker(
                                time_sec=t_sec,
                                name=_format_marker_name(f"ピッチ確認 ({c_off:+.0f}ct)", lyrics_alignment, t_sec),
                                category="pitch",
                            )
                        )
        final_vocal = aligned_vocal
    else:
        if slipped_cnt > 0:
            f0, valid_mask, med_f0, med_cents = analyze_vocal_f0_and_precision(aligned_vocal, sr=sr)
        final_vocal, nudged_cnt, pitch_markers = apply_td_psola_nuance_pitch_correction(
            aligned_vocal,
            sr=sr,
            f0=f0,
            valid_mask=valid_mask,
            scale_mask=scale_mask,
            strength=0.82 if force_pitch else 0.50,
            lyrics_alignment=lyrics_alignment,
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


def evaluate_raw_vocal_correction_benchmark(
    reference_tuned_vocal: np.ndarray,
    sr: int,
    scale_mask: np.ndarray,
) -> dict:
    """Quantitatively benchmark raw vocal pitch correction + consonant attack preservation."""
    from scipy import signal as sp_signal

    ref_mono = np.mean(reference_tuned_vocal, axis=0) if reference_tuned_vocal.ndim == 2 else reference_tuned_vocal
    ref_mono = np.asarray(ref_mono, dtype=np.float32)

    ref_align = align_lyrics_and_phonemes(ref_mono, sr=sr, lyrics_path=None)
    hop = 512
    f0, valid_mask, _, _ = analyze_vocal_f0_and_precision(ref_mono, sr=sr)
    frame_protect = _map_consonant_protect_to_frames(ref_align, len(f0), hop=hop, sr=sr)

    # Synthesize controlled untuned raw vocal: smoothly detune sustained vowels by +30 cents while locking consonants
    detune_ratio = float(2.0 ** (0.30 / 12.0))
    shift_frames = np.ones(len(f0), dtype=np.float64)
    labels, n_reg = ndimage.label(valid_mask & (~frame_protect))
    for r_i in range(1, n_reg + 1):
        idxs = np.where(labels == r_i)[0]
        if len(idxs) >= 4:
            ramp = np.ones(len(idxs), dtype=np.float64)
            edge = min(3, len(idxs) // 3)
            if edge > 1:
                ramp[:edge] = np.linspace(0.0, 1.0, edge)
                ramp[-edge:] = np.linspace(1.0, 0.0, edge)
            shift_frames[idxs] = 1.0 + (detune_ratio - 1.0) * ramp
    shift_frames[frame_protect] = 1.0

    untuned_vocal, _ = _resynth_pitch_tier_praat(
        ref_mono, sr=sr, hop=hop, shift_ratio_frames=shift_frames, lyrics_alignment=ref_align
    )

    untuned_align = align_lyrics_and_phonemes(untuned_vocal, sr=sr, lyrics_path=None)
    _, _, _, initial_cents_error = analyze_vocal_f0_and_precision(untuned_vocal, sr=sr)

    corrected_vocal, pt_rep = run_smart_pitch_and_timing_stage(
        untuned_vocal,
        sr=sr,
        scale_mask=scale_mask,
        force_pitch=True,
        lyrics_alignment=untuned_align,
    )
    _, _, _, corrected_cents_error = analyze_vocal_f0_and_precision(corrected_vocal, sr=sr)

    # Measure 1.8-4.8 kHz consonant-attack envelope preservation correlation
    nyq = 0.5 * sr
    hi = min(4800.0, nyq * 0.90)
    sos_atk = sp_signal.butter(4, [1800.0, hi], btype="bandpass", fs=sr, output="sos")
    padlen = min(len(untuned_vocal) - 1, 3 * (2 * sos_atk.shape[0] + 1))
    u_atk = sp_signal.sosfiltfilt(sos_atk, untuned_vocal.astype(np.float64), padlen=padlen)
    c_atk = sp_signal.sosfiltfilt(sos_atk, corrected_vocal.astype(np.float64), padlen=padlen)

    env_hop = max(64, int(round(sr * 0.01)))
    n_f = len(untuned_vocal) // env_hop
    if n_f >= 4:
        e_u = np.sqrt(np.mean(u_atk[: n_f * env_hop].reshape(n_f, env_hop) ** 2, axis=1) + 1e-12)
        e_c = np.sqrt(np.mean(c_atk[: n_f * env_hop].reshape(n_f, env_hop) ** 2, axis=1) + 1e-12)
        if np.std(e_u) > 1e-9 and np.std(e_c) > 1e-9:
            atk_corr = float(np.corrcoef(e_u, e_c)[0, 1])
        else:
            atk_corr = 1.0
    else:
        atk_corr = 1.0

    return {
        "initial_cents_error": round(float(initial_cents_error), 2),
        "corrected_cents_error": round(float(corrected_cents_error), 2),
        "cents_improvement": round(float(initial_cents_error - corrected_cents_error), 2),
        "consonant_attack_band_preservation_corr": round(float(np.clip(atk_corr, -1.0, 1.0)), 4),
        "notes_nudged_count": int(pt_rep.notes_nudged_count),
    }
