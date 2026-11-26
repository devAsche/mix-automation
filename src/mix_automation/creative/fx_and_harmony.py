"""v2 Stage 3 & Stage 4 Creative Engine: Reference-Verified FX & Nectar 3 Chord-Safe Harmonies.

Solves every v1 creative flaw identified during listening evaluation:
  1. Stage ③ Strict Reference Evidence Gate:
     - Eliminates v1's blind pause-based reverb throws that splashed loud tails onto dry breaks.
     - Compares the separated original reference vocal (`ref_vocal_mono`) against the user's vocal
       at phrase endings: ONLY triggers `ethereal_reverb_throw` when the original mix genuinely
       exhibits a prominent lingering wet tail (>= +6.5 dB above the song's baseline decay without
       dry syllable re-attacks), rendered chorus-free at a tasteful <= -15 dB relative level.
     - Detects `radio_voice` by comparing local bandpass concentration against the original vocal's
       own song-wide spectral median, preventing false positives on mid-forward singers.
  2. Stage ④ Genuine `iZotope Nectar 3` Chord-Safe Harmony Engine:
     - Loads commercial `Nectar 3.vst3` (`harmony_solo_harmony=True`, `pitch_formant_preservation=True`,
       `pitch_scale='Chromatic'`, non-harmony modules bypassed) to synthesize harmonies in the
       singer's authentic formant timbre.
     - Eliminates scale/modulation/borrowed-chord clashes by scoring candidate chromatic intervals
       (`+3`, `+4`, `+5` for Upper; `-3`, `-4`, `-5` for Lower) per sung note against the backing
       track's local CQT chromagram + local sliding key mask, switching intervals strictly at
       consonant/note boundaries with zero mid-vowel pitch jumps.
     - Outputs healthy `-18 LUFS` stems for Studio One 5 (fader pre-set to ~`-10.5 dB` L/R so they
       work both as a turn-key final mix out of the box and as soloable recording guides) plus
       ducked mix-ready stereo buses where user harmonies are absent!
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import librosa
import mido
import numpy as np
import psola
import pyloudnorm as pyln
from pedalboard import (
    Compressor,
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
    Pedalboard,
    Reverb,
    load_plugin,
)
from scipy import ndimage, signal

from mix_automation.config.plugin_registry import get_plugin_map
from mix_automation.dsp.lyrics_aligner import find_lyric_snippet_at_span
from mix_automation.dsp.vst3_adapter import (
    configure_khs_dual_delay_spot,
    configure_mautopitch_hard_tune,
    configure_trash2_vocal_grit,
    configure_vocalsynth2_radio,
)
from mix_automation.dsp.vst3_optimizer import SongSection


@dataclass(frozen=True)
class CreativeFXEvent:
    """Reference-verified localized vocal effect cue on the timeline."""

    fx_type: str  # "ethereal_reverb_throw" | "radio_voice" | "hard_autotune" | "spot_delay_throw" | "distortion_grit"
    start_sec: float
    end_sec: float
    confidence: float
    ref_evidence_db: float
    reason: str
    lyric_snippet: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HarmonySuggestion:
    """Director's section-by-section chorus & harmony arrangement recommendation."""

    section_label: str
    start_sec: float
    end_sec: float
    arrangement_role: str
    recommended_layers: list[str]
    director_note_ja: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HarmonyGenerationResult:
    """Complete output of the Nectar 3 chord-safe harmony generator."""

    upper_guide_mono: np.ndarray        # Staged -18 LUFS mono stem for Studio One track (pan L, fader -10.5 dB)
    lower_guide_mono: np.ndarray        # Staged -18 LUFS mono stem for Studio One track (pan R, fader -11.0 dB)
    auto_mix_harmony_stereo: np.ndarray # Pre-panned, EQ-pocketed & user-harmony-ducked stereo bus for 2mix
    engine_used: str                    # "izotope_nectar3_vst3" | "td_psola_fallback"
    total_harmony_notes: int
    chord_clash_notes_rescued: int
    suggestions: list[HarmonySuggestion]
    counter_melody_notes: list[dict] = field(default_factory=list)
    midi_guide_path: str | None = None


def _echo_autocorr_peak(tail_env: np.ndarray, lag_frames: int) -> float:
    """Compute zero-mean normalized autocorrelation peak near `lag_frames` with rebound check."""
    if lag_frames < 2 or len(tail_env) < lag_frames + 8:
        return 0.0
    win = max(5, min(lag_frames - 1, 15))
    trend = ndimage.uniform_filter1d(tail_env.astype(np.float64), size=win)
    hp = tail_env.astype(np.float64) - trend
    best = 0.0
    for lag in range(max(2, lag_frames - 2), min(len(hp) - 4, lag_frames + 3)):
        a = hp[:-lag]
        b = hp[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        c = float(np.dot(a, b)) / denom
        if c > best:
            best = c
    return best


def detect_creative_fx_events(
    lead_vocal_mono: np.ndarray,
    inst_mono: np.ndarray,
    sr: int,
    ref_vocal_mono: np.ndarray | None = None,
    lyrics_alignment: Any | None = None,
    bpm: float = 120.0,
) -> list[CreativeFXEvent]:
    """Detect multi-class Stage-3 FX cues STRICTLY verified against original reference vocal."""
    if ref_vocal_mono is None or len(ref_vocal_mono) < sr * 8:
        # Without original vocal evidence, never hallucinate arbitrary throws!
        return []

    events: list[CreativeFXEvent] = []
    n = min(len(lead_vocal_mono), len(ref_vocal_mono))
    hop = int(sr * 0.02)  # 20ms frames
    n_frames = n // hop
    if n_frames < 50:
        return events

    u_framed = lead_vocal_mono[: n_frames * hop].reshape(n_frames, hop)
    r_framed = ref_vocal_mono[: n_frames * hop].reshape(n_frames, hop)

    u_rms_lin = np.sqrt(np.mean(u_framed**2, axis=1) + 1e-12)
    r_rms_lin = np.sqrt(np.mean(r_framed**2, axis=1) + 1e-12)
    u_rms_db = 20.0 * np.log10(u_rms_lin)
    r_rms_db = 20.0 * np.log10(r_rms_lin)

    u_peak_db = float(np.percentile(u_rms_db, 96))
    r_peak_db = float(np.percentile(r_rms_db, 96))
    u_active = u_rms_db >= (u_peak_db - 24.0)
    r_active = r_rms_db >= (r_peak_db - 26.0)

    # 1. Reference-Verified Ethereal Reverb Throw & Rhythmic Spot Delay Throw Detection
    labels, n_phrases = ndimage.label(u_active)
    candidate_throws: list[tuple[int, int, float, float]] = []
    all_valid_rest_tails_db: list[float] = []
    delay_throw_end_frames: set[int] = set()

    lag_eighth_f = max(4, int(round((60.0 / max(bpm, 60.0)) * 0.5 / 0.02)))
    lag_quarter_f = max(8, int(round((60.0 / max(bpm, 60.0)) / 0.02)))

    for p_idx in range(1, n_phrases + 1):
        idxs = np.where(labels == p_idx)[0]
        if len(idxs) * 0.02 < 0.40:
            continue
        end_f = int(idxs[-1])
        look_end = min(n_frames, end_f + int(1.50 / 0.02))
        gap_f = 0
        for f in range(end_f + 1, look_end):
            if not u_active[f]:
                gap_f += 1
            else:
                break
        gap_sec = gap_f * 0.02
        if gap_sec < 0.55:
            continue

        pre_s = max(0, end_f - 20)
        if not np.any(r_active[pre_s : end_f + 2]):
            continue
        ref_pre_db = float(np.percentile(r_rms_db[pre_s : end_f + 1], 80))

        # 1a. Rhythmic Spot Delay Throw check (1/8 or 1/4 beat echo pulses in post-phrase rest)
        d_tail_s = min(n_frames, end_f + 1)
        d_tail_e = min(n_frames, end_f + gap_f)
        if d_tail_e - d_tail_s >= max(16, lag_eighth_f + 8):
            d_slice_lin = r_rms_lin[d_tail_s:d_tail_e]
            d_slice_db = r_rms_db[d_tail_s:d_tail_e]
            d_rel_db = float(np.mean(d_slice_db)) - ref_pre_db
            # Require audible tail energy above noise floor without sustained dry singing
            if -36.0 <= d_rel_db <= -5.5:
                ac_8 = _echo_autocorr_peak(d_slice_lin, lag_eighth_f)
                ac_4 = _echo_autocorr_peak(d_slice_lin, lag_quarter_f)
                best_ac = max(ac_8, ac_4)
                win_hp = max(5, min(lag_eighth_f - 1, 15))
                hp_tail = d_slice_lin.astype(np.float64) - ndimage.uniform_filter1d(
                    d_slice_lin.astype(np.float64), size=win_hp
                )
                mod_depth = float(np.std(hp_tail) / (np.mean(d_slice_lin) + 1e-9))
                if best_ac >= 0.42 and mod_depth >= 0.08:
                    subdiv = "1/8" if ac_8 >= ac_4 else "1/4"
                    t0 = round(max(int(idxs[0]) * 0.02, end_f * 0.02 - 0.34), 2)
                    t1 = round(end_f * 0.02 + 0.05, 2)
                    delay_throw_end_frames.add(end_f)
                    events.append(
                        CreativeFXEvent(
                            fx_type="spot_delay_throw",
                            start_sec=t0,
                            end_sec=t1,
                            confidence=round(float(np.clip(best_ac, 0.45, 0.96)), 2),
                            ref_evidence_db=round(d_rel_db + 30.0, 2),
                            reason=f"本家休符({gap_sec:.2f}秒)で{subdiv}拍周期ディレイ反復(r={best_ac:.2f})を検出",
                            lyric_snippet=find_lyric_snippet_at_span(lyrics_alignment, t0, t1),
                        )
                    )
                    continue

        if gap_sec < 0.65:
            continue

        # 1b. Diffuse Ethereal Reverb Throw check
        tail_s = min(n_frames, end_f + 4)   # +80ms after phrase offset
        tail_e = min(n_frames, end_f + 26)  # up to +520ms into the vocal rest
        if tail_e - tail_s < 10:
            continue

        ref_tail_slice = r_rms_db[tail_s:tail_e]
        ref_tail_db = float(np.mean(ref_tail_slice))

        # Reject if the original singer starts a new dry syllable/ad-lib during the rest (upward attack > +4.5 dB)
        ref_diffs = np.diff(ref_tail_slice)
        if len(ref_diffs) > 0 and float(np.max(ref_diffs)) > 4.5:
            continue
        # Also require overall decaying trend (first half of tail >= second half of tail)
        half = len(ref_tail_slice) // 2
        if float(np.mean(ref_tail_slice[half:])) > float(np.mean(ref_tail_slice[:half])) + 1.2:
            continue

        tail_rel_db = ref_tail_db - ref_pre_db
        # If tail_rel_db > -6.0 dB, the original singer is still sustaining a dry note rather than a reverb tail
        if tail_rel_db <= -6.0:
            all_valid_rest_tails_db.append(max(tail_rel_db, -36.0))
            if tail_rel_db >= -34.0 and end_f not in delay_throw_end_frames:
                candidate_throws.append((int(idxs[0]), end_f, gap_sec, tail_rel_db))

    if candidate_throws and all_valid_rest_tails_db:
        baseline_tail_db = float(np.median(all_valid_rest_tails_db))
        # Require at least +7.5 dB stronger lingering tail than the song's normal room/plate baseline AND >= -24.5 dB
        trigger_thresh_db = max(baseline_tail_db + 7.5, -24.5)
        verified_throws: list[CreativeFXEvent] = []
        for start_f, end_f, gap_sec, tail_rel_db in candidate_throws:
            excess_db = tail_rel_db - baseline_tail_db
            if tail_rel_db >= trigger_thresh_db:
                t0 = round(max(start_f * 0.02, end_f * 0.02 - 0.34), 2)
                t1 = round(end_f * 0.02 + 0.05, 2)
                conf = round(float(np.clip((excess_db - 5.0) / 8.0, 0.35, 0.98)), 2)
                verified_throws.append(
                    CreativeFXEvent(
                        fx_type="ethereal_reverb_throw",
                        start_sec=t0,
                        end_sec=t1,
                        confidence=conf,
                        ref_evidence_db=round(excess_db, 2),
                        reason=(
                            f"本家ボーカルの該当休符（{gap_sec:.2f}秒）にて通常比 +{excess_db:.1f}dB の長い空間余韻を検出"
                        ),
                        lyric_snippet=find_lyric_snippet_at_span(lyrics_alignment, t0, t1),
                    )
                )
        # Keep at most top 6 highest-confidence genuine moments so throws stay special
        verified_throws.sort(key=lambda e: e.confidence, reverse=True)
        events.extend(sorted(verified_throws[:6], key=lambda e: e.start_sec))

    # 2. Reference-Verified Radio / Telephone Band-Limited Vocal & Distortion Grit Detection
    freqs, _, Z_ref = signal.stft(ref_vocal_mono[:n], fs=sr, nperseg=2048, noverlap=1024)
    P_ref = np.abs(Z_ref) ** 2
    low_m = (freqs >= 95) & (freqs < 330)
    mid_m = (freqs >= 700) & (freqs <= 2800)
    hi_m = (freqs >= 5000) & (freqs <= 12500)

    low_e = np.sum(P_ref[low_m], axis=0) + 1e-12
    mid_e = np.sum(P_ref[mid_m], axis=0) + 1e-12
    hi_e = np.sum(P_ref[hi_m], axis=0) + 1e-12
    tot_db = 10.0 * np.log10(low_e + mid_e + hi_e)
    act_stft = tot_db > (np.percentile(tot_db, 95) - 20.0)
    hop_stft = 1024 / sr

    radio_mask = np.zeros_like(act_stft, dtype=bool)
    if np.sum(act_stft) >= 30:
        ml_db = 10.0 * np.log10(mid_e / low_e)
        mh_db = 10.0 * np.log10(mid_e / hi_e)
        med_ml = float(np.median(ml_db[act_stft]))
        med_mh = float(np.median(mh_db[act_stft]))

        # Must deviate sharply (+9.5 dB simultaneously in low-cut AND high-cut) from the singer's own normal tone!
        radio_mask = (
            act_stft
            & (ml_db > max(15.0, med_ml + 9.5))
            & (mh_db > max(15.0, med_mh + 9.5))
        )
        r_labels, r_num = ndimage.label(ndimage.binary_closing(radio_mask, structure=np.ones(6)))
        for r_i in range(1, r_num + 1):
            r_idxs = np.where(r_labels == r_i)[0]
            if len(r_idxs) * hop_stft >= 0.95:
                t0 = round(float(r_idxs[0] * hop_stft), 2)
                t1 = round(float((r_idxs[-1] + 1) * hop_stft), 2)
                ev_db = round(float(np.mean(ml_db[r_idxs] - med_ml)), 2)
                events.append(
                    CreativeFXEvent(
                        fx_type="radio_voice",
                        start_sec=t0,
                        end_sec=t1,
                        confidence=0.86,
                        ref_evidence_db=ev_db,
                        reason=f"本家ボーカルにて通常比 +{ev_db:.1f}dB の帯域制限（ラジオ/電話ボイス演出）を検出",
                        lyric_snippet=find_lyric_snippet_at_span(lyrics_alignment, t0, t1),
                    )
                )

        # 2b. Distortion Grit Detection (requires BOTH inter-harmonic noise/clipping flatness jump AND saturation jump,
        # preventing clean Vocaloid high-register chorus belts like GUMI/Flower from triggering false distortion!)
        grit_m = (freqs >= 1200.0) & (freqs <= 4200.0)
        fund_m = (freqs >= 180.0) & (freqs <= 650.0)
        if np.any(grit_m) and np.any(fund_m):
            P_g = P_ref[grit_m] + 1e-12
            flatness_db = 10.0 * np.log10(
                np.exp(np.mean(np.log(P_g), axis=0)) / (np.mean(P_g, axis=0) + 1e-12)
            )
            sat_ratio_db = 10.0 * np.log10(
                (np.sum(P_ref[grit_m], axis=0) + 1e-12) / (np.sum(P_ref[fund_m], axis=0) + 1e-12)
            )
            grit_score_db = flatness_db + sat_ratio_db
            med_flat = float(np.median(flatness_db[act_stft]))
            med_sat = float(np.median(sat_ratio_db[act_stft]))
            med_grit = float(np.median(grit_score_db[act_stft]))
            grit_mask = (
                act_stft
                & (~radio_mask)
                & (flatness_db >= -5.5)
                & ((flatness_db - med_flat) >= 3.8)
                & ((sat_ratio_db - med_sat) >= 4.0)
                & ((grit_score_db - med_grit) >= 8.5)
            )
            g_labels, g_num = ndimage.label(ndimage.binary_closing(grit_mask, structure=np.ones(5)))
            for g_i in range(1, g_num + 1):
                g_idxs = np.where(g_labels == g_i)[0]
                if len(g_idxs) * hop_stft >= 1.00:
                    t0 = round(float(g_idxs[0] * hop_stft), 2)
                    t1 = round(float((g_idxs[-1] + 1) * hop_stft), 2)
                    ev_db = round(float(np.mean(grit_score_db[g_idxs] - med_grit)), 2)
                    events.append(
                        CreativeFXEvent(
                            fx_type="distortion_grit",
                            start_sec=t0,
                            end_sec=t1,
                            confidence=round(float(np.clip(0.55 + (ev_db - 8.5) * 0.04, 0.60, 0.94)), 2),
                            ref_evidence_db=ev_db,
                            reason=f"本家ボーカルにて 1.2-4.2kHz 非整数倍音歪み/クリッピング +{ev_db:.1f}dB を検出",
                            lyric_snippet=find_lyric_snippet_at_span(lyrics_alignment, t0, t1),
                        )
                    )

    # 3. Reference-Verified Hard Auto-Tune Detection (sub-cent contour: robotic < 2.6ct + sharp step vs natural > 6.5ct)
    dt_at = 0.02
    r_f0: np.ndarray | None = None
    u_f0: np.ndarray | None = None
    try:
        import parselmouth

        r_snd = parselmouth.Sound(ref_vocal_mono[:n].astype(np.float64), sampling_frequency=float(sr))
        u_snd = parselmouth.Sound(lead_vocal_mono[:n].astype(np.float64), sampling_frequency=float(sr))
        r_p = r_snd.to_pitch(time_step=dt_at, pitch_floor=75.0, pitch_ceiling=900.0)
        u_p = u_snd.to_pitch(time_step=dt_at, pitch_floor=75.0, pitch_ceiling=900.0)
        r_f0 = np.asarray(r_p.selected_array["frequency"], dtype=np.float64)
        u_f0 = np.asarray(u_p.selected_array["frequency"], dtype=np.float64)
        n_at = min(len(r_f0), len(u_f0))
        r_f0, u_f0 = r_f0[:n_at], u_f0[:n_at]
        r_valid = r_f0 > 0.0
        u_valid = u_f0 > 0.0
    except Exception:
        pyin_hop = max(256, int(round(sr * dt_at)))
        r_f0, r_vf, r_vp = librosa.pyin(ref_vocal_mono[:n], fmin=75.0, fmax=900.0, sr=sr, hop_length=pyin_hop)
        u_f0, u_vf, u_vp = librosa.pyin(lead_vocal_mono[:n], fmin=75.0, fmax=900.0, sr=sr, hop_length=pyin_hop)
        n_at = min(len(r_f0), len(u_f0))
        r_f0, u_f0 = r_f0[:n_at], u_f0[:n_at]
        r_valid = (~np.isnan(r_f0)) & r_vf[:n_at] & (r_vp[:n_at] > 0.65)
        u_valid = (~np.isnan(u_f0)) & u_vf[:n_at] & (u_vp[:n_at] > 0.65)

    if r_f0 is not None and u_f0 is not None and np.sum(r_valid) >= 24 and np.sum(u_valid) >= 16:
        r_midi = 69.0 + 12.0 * np.log2(np.where(r_valid, r_f0, 220.0) / 440.0)
        u_midi = 69.0 + 12.0 * np.log2(np.where(u_valid, u_f0, 220.0) / 440.0)
        r_diff = np.abs(np.diff(r_midi, prepend=r_midi[0])) * 100.0
        u_diff = np.abs(np.diff(u_midi, prepend=u_midi[0])) * 100.0

        min_len_f = int(round(0.85 / dt_at))
        bridge_f = max(2, int(round(0.12 / dt_at)))
        r_spans = ndimage.binary_closing(r_valid, structure=np.ones(bridge_f, dtype=bool))
        at_labels, at_num = ndimage.label(r_spans)
        for a_i in range(1, at_num + 1):
            a_idxs = np.where(at_labels == a_i)[0]
            if len(a_idxs) < min_len_f:
                continue
            f0_s, f1_s = int(a_idxs[0]), int(a_idxs[-1]) + 1
            r_consec = r_valid[f0_s + 1 : f1_s] & r_valid[f0_s : f1_s - 1]
            r_steady = r_consec & (r_diff[f0_s + 1 : f1_s] < 35.0)
            u_consec = u_valid[f0_s + 1 : f1_s] & u_valid[f0_s : f1_s - 1]
            u_steady = u_consec & (u_diff[f0_s + 1 : f1_s] < 45.0)
            if np.sum(r_steady) < 12 or np.sum(u_steady) < 8:
                continue

            med_r_ct = float(np.median(r_diff[f0_s + 1 : f1_s][r_steady]))
            med_u_ct = float(np.median(u_diff[f0_s + 1 : f1_s][u_steady]))

            # Sharp instantaneous semitone jumps (>= 80 ct within 1-3 frames) inside quantized phrase
            sharp_jump = bool(np.any(r_diff[f0_s + 1 : f1_s][r_consec] >= 80.0))
            if not sharp_jump:
                for step_k in (2, 3):
                    if f1_s - f0_s > step_k:
                        pair_v = r_valid[f0_s + step_k : f1_s] & r_valid[f0_s : f1_s - step_k]
                        j_ct = np.abs(r_midi[f0_s + step_k : f1_s] - r_midi[f0_s : f1_s - step_k]) * 100.0
                        if np.any(j_ct[pair_v] >= 85.0):
                            sharp_jump = True
                            break

            if med_r_ct < 2.6 and sharp_jump and med_u_ct > 6.5:
                t0 = round(float(f0_s * dt_at), 2)
                t1 = round(float(f1_s * dt_at), 2)
                events.append(
                    CreativeFXEvent(
                        fx_type="hard_autotune",
                        start_sec=t0,
                        end_sec=t1,
                        confidence=round(float(np.clip((med_u_ct - med_r_ct) / 10.0, 0.55, 0.96)), 2),
                        ref_evidence_db=round(med_u_ct - med_r_ct, 2),
                        reason=f"本家ピッチ微細変動 {med_r_ct:.2f}ct＋急峻ステップ遷移を検出（原声 {med_u_ct:.1f}ct）",
                        lyric_snippet=find_lyric_snippet_at_span(lyrics_alignment, t0, t1),
                    )
                )

    events.sort(key=lambda e: e.start_sec)
    return events


def _apply_hard_autotune_segment(
    seg: np.ndarray,
    sr: int,
    mauto_plug: Any | None,
    scale_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply MAutoPitch hard-tune (or robotic quantized TD-PSOLA fallback) to a vocal segment."""
    if mauto_plug is not None:
        try:
            st_seg = np.stack([seg, seg], axis=0).astype(np.float32)
            out_seg = np.mean(mauto_plug(st_seg, sr, reset=True), axis=0, dtype=np.float32)
            if float(np.sqrt(np.mean(out_seg**2))) > 1e-5:
                return out_seg
        except Exception:
            pass

    try:
        hop = max(128, int(sr / 100))
        f0, vf, _ = librosa.pyin(seg, fmin=75.0, fmax=850.0, sr=sr, hop_length=hop)
        valid = (~np.isnan(f0)) & vf
        if np.any(valid):
            midi = librosa.hz_to_midi(np.where(valid, f0, 220.0))
            if scale_mask is not None and len(scale_mask) == 12 and np.any(scale_mask):
                allowed = np.where(np.asarray(scale_mask, dtype=bool))[0]
                q_midi = np.empty_like(midi)
                for i, m in enumerate(midi):
                    octv = int(np.floor(m / 12.0)) * 12
                    cands = np.concatenate([allowed + octv - 12, allowed + octv, allowed + octv + 12])
                    q_midi[i] = cands[int(np.argmin(np.abs(cands - m)))]
            else:
                q_midi = np.round(midi)
            target_f0 = np.where(valid, librosa.midi_to_hz(q_midi), np.nan)
            return psola.vocode(
                seg.astype(np.float64),
                sample_rate=sr,
                target_pitch=target_f0,
                fmin=75.0,
                fmax=900.0,
            ).astype(np.float32)[: len(seg)]
    except Exception:
        pass
    return seg.astype(np.float32)


def apply_verified_inline_fx_chain(
    lead_mono: np.ndarray,
    sr: int,
    events: list[CreativeFXEvent],
    scale_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply Stage-3 inline FX (`radio_voice`, `hard_autotune`, `distortion_grit`) + return spot stems."""
    inline_types = {"radio_voice", "hard_autotune", "distortion_grit"}
    inline_events = [ev for ev in events if ev.fx_type in inline_types]
    if not inline_events:
        return lead_mono.astype(np.float32), {}

    out = lead_mono.copy().astype(np.float32)
    n = len(out)
    spot_stems: dict[str, np.ndarray] = {}
    pmap = get_plugin_map()

    vs_plug = None
    if any(ev.fx_type == "radio_voice" for ev in inline_events):
        if "izotope_vocalsynth2" in pmap and pmap["izotope_vocalsynth2"].exists:
            try:
                vs_plug = load_plugin(str(pmap["izotope_vocalsynth2"].loadable_path))
                configure_vocalsynth2_radio(vs_plug, freq_hz=1650.0, drive=3.8, distort_mix=42.0)
            except Exception:
                vs_plug = None

    mauto_plug = None
    if any(ev.fx_type == "hard_autotune" for ev in inline_events):
        if "mautopitch" in pmap and pmap["mautopitch"].exists:
            try:
                mauto_plug = load_plugin(str(pmap["mautopitch"].loadable_path))
                configure_mautopitch_hard_tune(mauto_plug, scale_mask=scale_mask, speed=95.0, depth=100.0)
            except Exception:
                mauto_plug = None

    trash_plug = None
    trash_board = None
    if any(ev.fx_type == "distortion_grit" for ev in inline_events):
        if "izotope_trash2" in pmap and pmap["izotope_trash2"].exists:
            try:
                trash_plug = load_plugin(str(pmap["izotope_trash2"].loadable_path))
                configure_trash2_vocal_grit(
                    trash_plug, drive=3.6, style=0.5, wet_mix=55.0, low_cut_hz=380.0, high_cut_hz=4200.0
                )
                trash_board = Pedalboard(
                    [
                        HighpassFilter(cutoff_frequency_hz=380.0),
                        trash_plug,
                        LowpassFilter(cutoff_frequency_hz=4200.0),
                    ]
                )
            except Exception:
                trash_plug = None
                trash_board = None

    radio_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=520.0),
            PeakFilter(cutoff_frequency_hz=1650.0, gain_db=3.2, q=1.25),
            LowpassFilter(cutoff_frequency_hz=3150.0),
            Compressor(threshold_db=-20.0, ratio=4.5, attack_ms=4.0, release_ms=70.0),
        ]
    )
    grit_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=380.0),
            Distortion(drive_db=11.0),
            LowpassFilter(cutoff_frequency_hz=4200.0),
            Gain(gain_db=-3.0),
        ]
    )

    for ev in inline_events:
        s0 = max(0, int(ev.start_sec * sr))
        s1 = min(n, int(ev.end_sec * sr))
        if s1 - s0 < 1024:
            continue
        dry_seg = out[s0:s1]
        dry_rms = float(np.sqrt(np.mean(dry_seg**2) + 1e-12))
        if dry_rms < 1e-4:
            continue

        fx_seg: np.ndarray
        target_scale = 0.96
        if ev.fx_type == "radio_voice":
            target_scale = 0.90
            if vs_plug is not None:
                st_seg = np.stack([dry_seg, dry_seg], axis=0).astype(np.float32)
                fx_seg = np.mean(vs_plug(st_seg, sr, reset=True), axis=0, dtype=np.float32)
                if float(np.sqrt(np.mean(fx_seg**2))) < 1e-5:
                    fx_seg = radio_board(dry_seg[np.newaxis, :].astype(np.float32), sr)[0]
            else:
                fx_seg = radio_board(dry_seg[np.newaxis, :].astype(np.float32), sr)[0]
        elif ev.fx_type == "hard_autotune":
            fx_seg = _apply_hard_autotune_segment(dry_seg, sr=sr, mauto_plug=mauto_plug, scale_mask=scale_mask)
        else:  # distortion_grit
            if trash_board is not None:
                st_seg = np.stack([dry_seg, dry_seg], axis=0).astype(np.float32)
                fx_seg = np.mean(trash_board(st_seg, sr), axis=0, dtype=np.float32)
                if float(np.sqrt(np.mean(fx_seg**2))) < 1e-5:
                    fx_seg = grit_board(dry_seg[np.newaxis, :].astype(np.float32), sr)[0]
            else:
                fx_seg = grit_board(dry_seg[np.newaxis, :].astype(np.float32), sr)[0]

        fx_seg = fx_seg[: len(dry_seg)]
        fx_rms = float(np.sqrt(np.mean(fx_seg**2) + 1e-12))
        fx_seg = (fx_seg * min((dry_rms * target_scale) / fx_rms, 2.5)).astype(np.float32)

        xf = min(int(0.04 * sr), len(dry_seg) // 4)
        w = np.ones(len(dry_seg), dtype=np.float32)
        if xf > 4:
            ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, xf))).astype(np.float32)
            w[:xf] = ramp
            w[-xf:] = ramp[::-1]

        out[s0:s1] = dry_seg * (1.0 - w) + fx_seg * w
        if ev.fx_type not in spot_stems:
            spot_stems[ev.fx_type] = np.zeros(n, dtype=np.float32)
        spot_stems[ev.fx_type][s0:s1] = (fx_seg * w).astype(np.float32)

    return out.astype(np.float32), spot_stems


def apply_verified_radio_fx_inline(
    lead_mono: np.ndarray,
    sr: int,
    events: list[CreativeFXEvent],
) -> np.ndarray:
    """Backwards-compatible wrapper applying smooth 40ms-crossfaded radio FX at `radio_voice` spans."""
    radio_events = [ev for ev in events if ev.fx_type == "radio_voice"]
    if not radio_events:
        return lead_mono
    processed, _ = apply_verified_inline_fx_chain(lead_mono, sr=sr, events=radio_events)
    return processed


def render_spot_delay_throw_stem(
    processed_lead_mono: np.ndarray,
    sr: int,
    events: list[CreativeFXEvent],
    bpm: float = 120.0,
) -> np.ndarray:
    """Render rhythmic stereo spot-delay throw stem ONLY at verified `spot_delay_throw` cues."""
    n = len(processed_lead_mono)
    delay_events = [ev for ev in events if ev.fx_type == "spot_delay_throw"]
    if not delay_events:
        return np.zeros((2, n), dtype=np.float32)

    source_feed = np.zeros(n, dtype=np.float32)
    active_samples = processed_lead_mono[np.abs(processed_lead_mono) > 1e-4]
    lead_rms = float(np.sqrt(np.mean(active_samples**2) + 1e-12)) if active_samples.size > 0 else 0.05

    for ev in delay_events:
        s0 = max(0, int(ev.start_sec * sr))
        s1 = min(n, int(ev.end_sec * sr))
        if s1 - s0 < 512:
            continue
        seg = processed_lead_mono[s0:s1].copy()
        fade = min(int(0.04 * sr), len(seg) // 3)
        if fade > 4:
            seg[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        source_feed[s0:s1] += seg * float(0.55 + 0.30 * ev.confidence)

    if np.max(np.abs(source_feed)) < 1e-5:
        return np.zeros((2, n), dtype=np.float32)

    is_quarter = any("1/4" in ev.reason for ev in delay_events)
    beat_mult = 1.0 if is_quarter else 0.5
    delay_ms = float(np.clip((60.0 / max(bpm, 60.0)) * 1000.0 * beat_mult, 120.0, 680.0))

    pmap = get_plugin_map()
    st_feed = np.stack([source_feed, source_feed], axis=0)
    wet_out: np.ndarray | None = None

    if "khs_dual_delay" in pmap and pmap["khs_dual_delay"].exists:
        try:
            dly = load_plugin(str(pmap["khs_dual_delay"].loadable_path))
            configure_khs_dual_delay_spot(dly, delay_ms=delay_ms, feedback=36.0, spread=75.0, duck=25.0, mix_pct=100.0)
            board = Pedalboard(
                [
                    HighpassFilter(cutoff_frequency_hz=340.0),
                    LowpassFilter(cutoff_frequency_hz=5400.0),
                    dly,
                ]
            )
            wet_out = board(st_feed, sr)
        except Exception:
            wet_out = None

    if wet_out is None:
        board = Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=340.0),
                LowpassFilter(cutoff_frequency_hz=5400.0),
                Delay(delay_seconds=delay_ms / 1000.0, feedback=0.36, mix=1.0),
            ]
        )
        wet_out = board(st_feed, sr)

    # Tastefully cap spot-delay throws at <= -14.5 dB relative to lead vocal!
    max_allowed_rms = max(lead_rms * (10.0 ** (-14.5 / 20.0)), 1e-4)
    wet_act = wet_out[np.abs(wet_out) > 1e-5]
    cur_rms = float(np.sqrt(np.mean(wet_act**2) + 1e-12)) if wet_act.size > 0 else 1e-6
    if cur_rms > max_allowed_rms:
        wet_out = wet_out * (max_allowed_rms / cur_rms)
    return wet_out.astype(np.float32)


_VS2_GLOSS_SUBPROCESS_SCRIPT = """
import json, sys
import numpy as np
from pedalboard import load_plugin
from mix_automation.dsp.vst3_adapter import configure_vocalsynth2_biovox_gloss

cfg = json.loads(sys.argv[1])
harm = np.load(cfg["in_npy"])
vs = load_plugin(cfg["vst_path"])
configure_vocalsynth2_biovox_gloss(vs, wet_mix=float(cfg["wet_mix"]), breath=16.0, clarity=76.0)
sr = int(cfg["sr"])
chunk = sr * 30
n = harm.shape[1]
out = np.zeros_like(harm, dtype=np.float32)
for s0 in range(0, n, chunk):
    s1 = min(n, s0 + chunk)
    out[:, s0:s1] = vs(harm[:, s0:s1].astype(np.float32), sr, reset=(s0 == 0)).astype(np.float32)[:, : s1 - s0]
np.save(cfg["out_npy"], out)
"""


def apply_vocalsynth2_harmony_gloss(
    harm_stereo: np.ndarray,
    sr: int,
    wet_mix: float = 12.0,
) -> np.ndarray:
    """Polish synthetic harmonies with licensed iZotope VocalSynth 2 Biovox in an isolated subprocess."""
    pmap = get_plugin_map()
    if wet_mix <= 0.5 or "izotope_vocalsynth2" not in pmap or not pmap["izotope_vocalsynth2"].exists:
        return harm_stereo
    try:
        with tempfile.TemporaryDirectory(prefix="automix_vs2_") as tmp_d:
            td = Path(tmp_d)
            in_p = td / "in.npy"
            out_p = td / "out.npy"
            np.save(str(in_p), harm_stereo.astype(np.float32))
            payload = json.dumps(
                {
                    "vst_path": str(pmap["izotope_vocalsynth2"].loadable_path),
                    "in_npy": str(in_p),
                    "out_npy": str(out_p),
                    "sr": int(sr),
                    "wet_mix": float(wet_mix),
                }
            )
            proc = subprocess.run(
                [sys.executable, "-c", "import json\n" + _VS2_GLOSS_SUBPROCESS_SCRIPT, payload],
                check=False,
                capture_output=True,
                timeout=90,
            )
            if proc.returncode == 0 and out_p.exists():
                y = np.load(str(out_p))
                if float(np.max(np.abs(y))) > 1e-5:
                    return y.astype(np.float32)
    except Exception:
        pass
    return harm_stereo


def render_ethereal_throw_stem(
    processed_lead_mono: np.ndarray,
    sr: int,
    events: list[CreativeFXEvent],
    bpm: float = 120.0,
) -> np.ndarray:
    """Render tasteful, chorus-free stereo throw stem ONLY at reference-verified throw cues."""
    n = len(processed_lead_mono)
    source_feed = np.zeros(n, dtype=np.float32)

    active_samples = processed_lead_mono[np.abs(processed_lead_mono) > 1e-4]
    lead_rms = float(np.sqrt(np.mean(active_samples**2) + 1e-12)) if active_samples.size > 0 else 0.05

    throw_events = [ev for ev in events if ev.fx_type == "ethereal_reverb_throw"]
    if not throw_events:
        return np.zeros((2, n), dtype=np.float32)

    for ev in throw_events:
        s0 = max(0, int(ev.start_sec * sr))
        s1 = min(n, int(ev.end_sec * sr))
        if s1 - s0 < 512:
            continue
        seg = processed_lead_mono[s0:s1].copy()
        fade = min(int(0.06 * sr), len(seg) // 3)
        if fade > 4:
            seg[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        source_feed[s0:s1] += seg * float(0.45 + 0.35 * ev.confidence)

    if np.max(np.abs(source_feed)) < 1e-5:
        return np.zeros((2, n), dtype=np.float32)

    pmap = get_plugin_map()
    st_feed = np.stack([source_feed, source_feed], axis=0)
    quarter_sec = float(np.clip(60.0 / max(bpm, 60.0), 0.25, 0.65))

    wet_out: np.ndarray | None = None
    if pmap["phoenix_verb"].exists:
        try:
            rv = load_plugin(str(pmap["phoenix_verb"].loadable_path))
            if "mix" in rv.parameters:
                rv.mix = 100.0
            board = Pedalboard(
                [
                    HighpassFilter(cutoff_frequency_hz=380.0),
                    LowpassFilter(cutoff_frequency_hz=6500.0),
                    Delay(delay_seconds=quarter_sec * 0.5, feedback=0.22, mix=0.35),
                    rv,
                ]
            )
            wet_out = board(st_feed, sr)
        except Exception:
            wet_out = None

    if wet_out is None:
        board = Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=380.0),
                LowpassFilter(cutoff_frequency_hz=6500.0),
                Delay(delay_seconds=quarter_sec * 0.5, feedback=0.22, mix=0.35),
                Reverb(room_size=0.74, damping=0.52, wet_level=1.0, dry_level=0.0, width=0.92),
            ]
        )
        wet_out = board(st_feed, sr)

    # Strictly cap throw peak so it sits as an ethereal -15.5 dB whisper behind the lead vocal!
    max_allowed_peak = max(lead_rms * (10.0 ** (-9.5 / 20.0)), 1e-4)
    cur_peak = float(np.max(np.abs(wet_out))) + 1e-9
    if cur_peak > max_allowed_peak:
        wet_out = wet_out * (max_allowed_peak / cur_peak)
    else:
        wet_out = wet_out * 0.18

    return wet_out.astype(np.float32)


def _build_section_arrangement_plan(
    sections: list[SongSection],
    n_samples: int,
    sr: int,
) -> tuple[np.ndarray, np.ndarray, list[HarmonySuggestion]]:
    """Create separate arrangement masks for Upper & Lower harmonies plus Director notes."""
    suggestions: list[HarmonySuggestion] = []
    upper_mask = np.zeros(n_samples, dtype=np.float32)
    lower_mask = np.zeros(n_samples, dtype=np.float32)
    chorus_seen = 0

    for sec in sections:
        s0 = max(0, int(sec.start_sec * sr))
        s1 = min(n_samples, int(sec.end_sec * sr))
        lbl = sec.label

        if "Instrumental" in lbl:
            suggestions.append(
                HarmonySuggestion(
                    section_label=lbl,
                    start_sec=sec.start_sec,
                    end_sec=sec.end_sec,
                    arrangement_role="間奏・イントロ",
                    recommended_layers=["息遣いアドリブ / フェイク（任意）"],
                    director_note_ja="ボーカル休符区間です。サビ直前の1小節にブレスやロングトーンフェイクを薄く入れると展開が際立ちます。",
                )
            )
        elif lbl.endswith("_Chorus"):
            chorus_seen += 1
            upper_mask[s0:s1] = 1.0
            if chorus_seen >= 2:
                lower_mask[s0:s1] = 0.90
                layers = ["メイン", "3度上ハモリ(Nectar補正済)", "3度下ハモリ(Nectar補正済)", "左右ダブル"]
                note = "後半・ラスサビの最高潮パートです。コード構成音に追従させた上下ハモリの両翼展開でサビの厚みと広がりを最大化しています。"
            else:
                # First chorus: primary upper harmony + second-half lower support
                mid_c = s0 + int((s1 - s0) * 0.5)
                lower_mask[mid_c:s1] = 0.70
                layers = ["メイン", "3度上ハモリ中心", "サビ後半から3度下ハモリ追加"]
                note = "1番サビです。前半は上ハモリを軸にスッキリ聴かせ、後半から下ハモリを足してラスサビへの伸びしろを残すプロ構成にしています。"
            suggestions.append(
                HarmonySuggestion(
                    section_label=lbl,
                    start_sec=sec.start_sec,
                    end_sec=sec.end_sec,
                    arrangement_role=f"サビ ({chorus_seen}回目)",
                    recommended_layers=layers,
                    director_note_ja=note,
                )
            )
        elif "Quiet" in lbl:
            tail_start = s0 + int((s1 - s0) * 0.72)
            upper_mask[tail_start:s1] = 0.65
            suggestions.append(
                HarmonySuggestion(
                    section_label=lbl,
                    start_sec=sec.start_sec,
                    end_sec=sec.end_sec,
                    arrangement_role="落ちサビ・静寂パート",
                    recommended_layers=["前半：メイン単体（ドライ寄り）", "終盤ビルドアップのみ上ハモリ"],
                    director_note_ja="オケが抜けるパートのため、前半はハモリを抜いて声の近さと息遣いを聴かせ、ラスサビ前の最後のビルドアップだけハモリを添えています。",
                )
            )
        else:
            # Verse / Pre-chorus: point harmony on last 35% (Bメロ build into chorus)
            build_pt = s0 + int((s1 - s0) * 0.65)
            upper_mask[build_pt:s1] = 0.68
            lower_mask[build_pt:s1] = 0.45
            suggestions.append(
                HarmonySuggestion(
                    section_label=lbl,
                    start_sec=sec.start_sec,
                    end_sec=sec.end_sec,
                    arrangement_role="平歌（Aメロ〜Bメロ）",
                    recommended_layers=["前半：メイン単体", "後半（Bメロ盛り上がり）：ポイントハモリ"],
                    director_note_ja="Aメロは単線で歌詞の輪郭を立たせ、サビへ向かうBメロ後半フレーズにハモリを薄く重ねて推進力を作っています。",
                )
            )

    fade_len = min(int(0.08 * sr), 4096)
    if fade_len > 4:
        win = np.hanning(fade_len).astype(np.float32)
        win /= np.sum(win)
        upper_mask = np.clip(signal.fftconvolve(upper_mask, win, mode="same"), 0.0, 1.0)
        lower_mask = np.clip(signal.fftconvolve(lower_mask, win, mode="same"), 0.0, 1.0)

    return upper_mask, lower_mask, suggestions


def _compute_sliding_chord_and_scale_profiles(
    inst_mono: np.ndarray,
    sr: int,
    hop: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame local chord chroma `(12, T)` and modulation-aware local scale `(12, T)`."""
    chroma = librosa.feature.chroma_cqt(y=inst_mono, sr=sr, hop_length=hop)
    # Smooth chord energy over ~0.32s window to suppress drum transients while tracking chord changes
    chord_win = max(3, int(0.32 * sr / hop))
    chord_chroma = ndimage.uniform_filter1d(chroma.astype(np.float32), size=chord_win, axis=1)
    max_per_f = np.max(chord_chroma, axis=0, keepdims=True) + 1e-8
    chord_chroma = chord_chroma / max_per_f

    # Sliding 8-second key window so final-chorus modulations (+1/+2 st) are tracked automatically!
    key_win = max(16, int(8.0 * sr / hop))
    key_chroma = ndimage.uniform_filter1d(chroma.astype(np.float32), size=key_win, axis=1)

    major_mask = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    minor_mask = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float32)
    templates = np.stack(
        [np.roll(major_mask, r) for r in range(12)] + [np.roll(minor_mask, r) for r in range(12)],
        axis=0,
    )  # (24, 12)

    scores = templates @ key_chroma  # (24, T)
    best_tpl_idx = np.argmax(scores, axis=0)
    local_scale = templates[best_tpl_idx].T  # (12, T)
    return chord_chroma, local_scale


_NECTAR3_SUBPROCESS_SCRIPT = """
import json, sys
from pathlib import Path
import numpy as np
from pedalboard import load_plugin

cfg = json.loads(sys.argv[1])
lead_mono = np.load(cfg["in_npy"])
sr = int(cfg["sr"])
intervals = cfg["intervals"]
out_dir = Path(cfg["out_dir"])

nectar = load_plugin(cfg["vst_path"])
for bp_key in (
    "gate_bypass", "eq_1_bypass", "eq_2_bypass", "comp_1_bypass", "comp_2_bypass",
    "deesser_bypass", "delay_bypass", "saturation_bypass", "reverb_bypass",
    "dimension_bypass", "alm_bypass", "limiter_bypass", "unmask_bypass",
):
    if bp_key in nectar.parameters:
        setattr(nectar, bp_key, True)

nectar.global_bypass = False
nectar.harmony_bypass = False
nectar.harmony_solo_harmony = True
nectar.harmony_wet_mix = 100.0
nectar.pitch_formant_preservation = True
nectar.pitch_scale = "Chromatic"
nectar.harmony_pitch_correction = 0.0
nectar.harmony_time_variation = 12.0
nectar.harmony_pitch_variation = 8.0
nectar.harmony_low_shelf_freq = 220.0
nectar.harmony_low_shelf_gain = -4.0
nectar.harmony_high_shelf_freq = 5200.0
nectar.harmony_high_shelf_gain = -2.0

for v_idx in range(1, 9):
    en_k = f"harmony_v{v_idx}_enable"
    if en_k in nectar.parameters:
        setattr(nectar, en_k, v_idx == 1)
nectar.harmony_v1_mute = False
nectar.harmony_v1_gain_db = 0.0
nectar.harmony_v1_pan = 0.0
nectar.harmony_v1_delay_ms = 5.0

n = len(lead_mono)
chunk = sr * 30
for st_int in intervals:
    nectar.harmony_v1_chrom_int = f"{int(st_int):+d}"
    buf = np.zeros(n, dtype=np.float32)
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        seg = lead_mono[s0:s1]
        if float(np.max(np.abs(seg))) < 1e-5:
            continue
        st_seg = np.stack([seg, seg], axis=0).astype(np.float32)
        y_st = nectar(st_seg, sr, reset=True)
        buf[s0:s1] = np.mean(y_st, axis=0, dtype=np.float32)[: s1 - s0]
    np.save(str(out_dir / f"cand_{int(st_int)}.npy"), buf)
"""


def _render_candidate_intervals_via_nectar3(
    lead_mono: np.ndarray,
    sr: int,
    intervals: tuple[int, ...],
    use_vst3: bool = True,
) -> tuple[dict[int, np.ndarray], str]:
    """Render candidate chromatic intervals (`+3`, `+4`, `+5`, `-3`, `-4`, `-5`) via isolated Nectar 3 subprocess."""
    import json

    pmap = get_plugin_map()
    candidates: dict[int, np.ndarray] = {}

    if use_vst3 and pmap["izotope_nectar3"].exists and len(lead_mono) >= sr * 2:
        try:
            with tempfile.TemporaryDirectory(prefix="automix_nectar3_") as tmp_d:
                td = Path(tmp_d)
                in_p = td / "lead.npy"
                np.save(str(in_p), lead_mono.astype(np.float32))
                payload = json.dumps(
                    {
                        "vst_path": str(pmap["izotope_nectar3"].loadable_path),
                        "in_npy": str(in_p),
                        "out_dir": str(td),
                        "sr": int(sr),
                        "intervals": [int(x) for x in intervals],
                    }
                )
                proc = subprocess.run(
                    [sys.executable, "-c", _NECTAR3_SUBPROCESS_SCRIPT, payload],
                    check=False,
                    capture_output=True,
                    timeout=180,
                )
                if proc.returncode == 0:
                    for st_int in intervals:
                        cp = td / f"cand_{int(st_int)}.npy"
                        if cp.exists():
                            candidates[int(st_int)] = np.load(str(cp)).astype(np.float32)[: len(lead_mono)]
                    if len(candidates) == len(intervals):
                        return candidates, "izotope_nectar3_vst3"
                candidates.clear()
        except Exception:
            candidates.clear()

    # Clean TD-PSOLA formant-preserving fallback (used for ultra-short clips or headless CI tests)
    for st_int in intervals:
        ratio = float(2.0 ** (st_int / 12.0))
        try:
            f0, _, _ = librosa.pyin(lead_mono, fmin=70.0, fmax=850.0, sr=sr, hop_length=int(sr / 100))
            target_f0 = np.where(np.isnan(f0), np.nan, f0 * ratio)
            shifted = psola.vocode(
                lead_mono.astype(np.float64),
                sample_rate=sr,
                target_pitch=target_f0,
                fmin=70.0,
                fmax=950.0,
            ).astype(np.float32)
            candidates[st_int] = shifted[: len(lead_mono)]
        except Exception:
            candidates[st_int] = np.zeros_like(lead_mono)

    return candidates, "td_psola_fallback"


def _extract_sung_note_spans(
    lead_f0: np.ndarray,
    voiced_mask: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Segment continuous F0 contour into discrete quantized sung notes `(start_f, end_f, midi_note)`."""
    n_frames = min(len(lead_f0), len(voiced_mask))
    midi_cont = librosa.hz_to_midi(np.where(np.isnan(lead_f0[:n_frames]), 60.0, lead_f0[:n_frames]))
    midi_round = np.round(midi_cont).astype(int)

    note_spans: list[tuple[int, int, int]] = []
    in_note = False
    start_f = 0
    cur_midi = 60

    for f in range(n_frames):
        if voiced_mask[f] and not np.isnan(lead_f0[f]):
            m = int(midi_round[f])
            if not in_note:
                in_note = True
                start_f = f
                cur_midi = m
            elif abs(m - cur_midi) >= 1 and (f - start_f) >= 3:
                note_spans.append((start_f, f, cur_midi))
                start_f = f
                cur_midi = m
        else:
            if in_note:
                if f - start_f >= 2:
                    note_spans.append((start_f, f, cur_midi))
                in_note = False
    if in_note and (n_frames - start_f) >= 2:
        note_spans.append((start_f, n_frames, cur_midi))
    return note_spans


def _score_chord_safe_interval(
    p_lead: int,
    c_vec: np.ndarray,
    s_vec: np.ndarray,
    allowed_intervals: tuple[int, ...],
    prev_choice: int,
) -> tuple[int, int, float, dict[int, float]]:
    """Score candidate harmonic intervals against local backing chord chroma & sliding scale."""
    naive_int = allowed_intervals[0]
    for cand_3rd in allowed_intervals[:2]:
        if s_vec[(p_lead + cand_3rd) % 12] > 0.5:
            naive_int = cand_3rd
            break

    best_int = naive_int
    best_score = -999.0
    scores_by_int: dict[int, float] = {}

    for cand_int in allowed_intervals:
        h_pc = (p_lead + cand_int) % 12
        chord_support = float(c_vec[h_pc])
        in_scale = float(s_vec[h_pc])

        adj_energy = max(float(c_vec[(h_pc - 1) % 12]), float(c_vec[(h_pc + 1) % 12]))
        semitone_clash = adj_energy * max(0.0, 1.0 - 0.75 * chord_support)
        tritone_clash = float(c_vec[(h_pc + 6) % 12]) * (0.35 if chord_support < 0.25 else 0.0)
        third_prior = 0.45 if abs(cand_int) in (3, 4) else 0.0
        continuity_bonus = 0.15 if cand_int == prev_choice else 0.0

        score = (
            2.65 * chord_support
            + 1.35 * in_scale
            - 3.10 * semitone_clash
            - tritone_clash
            + third_prior
            + continuity_bonus
        )
        scores_by_int[cand_int] = score
        if score > best_score:
            best_score = score
            best_int = cand_int

    return best_int, naive_int, best_score, scores_by_int


def _select_and_stitch_chord_safe_harmony(
    candidates: dict[int, np.ndarray],
    allowed_intervals: tuple[int, int, int],
    lead_f0: np.ndarray,
    voiced_mask: np.ndarray,
    chord_chroma: np.ndarray,
    local_scale: np.ndarray,
    sr: int,
    hop: int = 512,
) -> tuple[np.ndarray, int, int]:
    """Select the most consonant interval per sung note and crossfade strictly at note boundaries."""
    n_frames = min(len(lead_f0), chord_chroma.shape[1], local_scale.shape[1])
    n_samples = len(next(iter(candidates.values())))
    note_spans = _extract_sung_note_spans(lead_f0[:n_frames], voiced_mask[:n_frames])

    # Frame-level interval assignment (held constant across unvoiced gaps to avoid unnecessary switches)
    chosen_per_frame = np.full(n_frames, allowed_intervals[0], dtype=int)
    rescued_count = 0
    prev_choice = allowed_intervals[0]

    for idx_n, (f0_i, f1_i, lead_m) in enumerate(note_spans):
        p_lead = lead_m % 12
        c_vec = np.mean(chord_chroma[:, f0_i:f1_i], axis=1)
        s_vec = np.round(np.mean(local_scale[:, f0_i:f1_i], axis=1))

        best_int, naive_int, best_score, scores_by_int = _score_chord_safe_interval(
            p_lead=p_lead,
            c_vec=c_vec,
            s_vec=s_vec,
            allowed_intervals=allowed_intervals,
            prev_choice=prev_choice,
        )

        if best_int != naive_int and (best_score - scores_by_int.get(naive_int, best_score)) >= 0.55:
            rescued_count += 1

        # Extend assignment from end of previous note through this note so transitions happen in silence/consonants
        fill_start = note_spans[idx_n - 1][1] if idx_n > 0 else 0
        chosen_per_frame[fill_start:f1_i] = best_int
        prev_choice = best_int

    if note_spans:
        chosen_per_frame[note_spans[-1][1] :] = prev_choice

    # Smooth per-interval weight masks (~18ms crossfade strictly at note transitions)
    out = np.zeros(n_samples, dtype=np.float32)
    sample_idx = np.arange(n_samples)
    frame_centers = np.arange(n_frames) * hop

    for cand_int in allowed_intervals:
        w_f = (chosen_per_frame == cand_int).astype(np.float32)
        w_f = ndimage.gaussian_filter1d(w_f, sigma=1.4)
        w_s = np.interp(sample_idx, frame_centers, w_f).astype(np.float32)
        out += candidates[cand_int] * w_s

    return out.astype(np.float32), len(note_spans), rescued_count


def _nearest_midi_in_register(pitch_class: int, anchor_midi: int, lo: int = 50, hi: int = 84) -> int:
    """Map `pitch_class` (0..11) to the octave closest to `anchor_midi` clamped to `[lo, hi]`."""
    cands = [pitch_class + 12 * oct_i for oct_i in range(2, 9)]
    valid = [c for c in cands if lo <= c <= hi] or cands
    return min(valid, key=lambda c: abs(c - anchor_midi))


def _compute_counter_melody_notes(
    note_spans: list[tuple[int, int, int]],
    chord_chroma: np.ndarray,
    local_scale: np.ndarray,
    sections: list[SongSection],
    bpm: float,
    sr: int,
    hop: int,
) -> list[dict]:
    """Generate chord-safe Call & Response motifs in phrase-end rests + High-5th/Octave obligato pads."""
    if not note_spans or chord_chroma.shape[1] == 0:
        return []

    frame_dt = hop / float(sr)
    n_frames = chord_chroma.shape[1]
    eighth_sec = float(np.clip(60.0 / max(bpm, 60.0) * 0.5, 0.14, 0.42))
    counter_notes: list[dict] = []

    chorus_sections = [s for s in sections if s.label.endswith("_Chorus")]
    final_chorus_labels = {s.label for s in chorus_sections[1:]} if len(chorus_sections) >= 2 else {s.label for s in chorus_sections}

    def _find_section(t_sec: float) -> SongSection | None:
        for sec in sections:
            if sec.start_sec <= t_sec <= sec.end_sec:
                return sec
        return None

    # 1. Call & Response (追っかけコーラス) in Chorus / Pre-Chorus phrase-ending rests (>= 0.75s)
    for idx_n, (f0_i, f1_i, lead_m) in enumerate(note_spans):
        end_sec = f1_i * frame_dt
        next_start_sec = note_spans[idx_n + 1][0] * frame_dt if idx_n + 1 < len(note_spans) else n_frames * frame_dt
        gap_sec = next_start_sec - end_sec
        if gap_sec < 0.75:
            continue

        sec = _find_section(end_sec)
        if sec is None:
            continue
        is_chorus = sec.label.endswith("_Chorus")
        is_late_pre = "Verse_PreChorus" in sec.label and (end_sec >= sec.start_sec + 0.55 * (sec.end_sec - sec.start_sec))
        if not (is_chorus or is_late_pre):
            continue

        r0_f = min(n_frames - 1, f1_i + max(1, int(round(0.08 / frame_dt))))
        r1_f = min(n_frames, f1_i + max(3, int(round(min(gap_sec - 0.10, 0.85) / frame_dt))))
        if r1_f <= r0_f:
            continue

        c_vec = np.mean(chord_chroma[:, r0_f:r1_f], axis=1)
        s_vec = np.round(np.mean(local_scale[:, r0_f:r1_f], axis=1))
        # Rank pitch classes that belong to active backing chord & scale
        pc_scores = [(float(c_vec[pc]) + 0.45 * float(s_vec[pc]), pc) for pc in range(12)]
        pc_scores.sort(reverse=True)
        top_pcs = [pc for sc, pc in pc_scores[:4] if sc >= 0.55]
        if len(top_pcs) < 2:
            top_pcs = [pc for _, pc in pc_scores[:3]]

        motif_count = 3 if gap_sec >= max(1.05, eighth_sec * 4.2) else 2
        anchor = int(np.clip(lead_m + 3, 55, 79))
        motif_pitches: list[int] = []
        for k_m in range(motif_count):
            pc = top_pcs[k_m % len(top_pcs)]
            m_note = _nearest_midi_in_register(pc, anchor_midi=anchor - k_m)
            if motif_pitches and m_note == motif_pitches[-1] and len(top_pcs) > 1:
                m_note = _nearest_midi_in_register(top_pcs[(k_m + 1) % len(top_pcs)], anchor_midi=anchor)
            motif_pitches.append(m_note)

        cursor_sec = end_sec + 0.12
        for k_m, m_note in enumerate(motif_pitches):
            dur = eighth_sec * (1.5 if k_m == motif_count - 1 else 0.92)
            if cursor_sec + 0.12 >= next_start_sec - 0.06:
                break
            n_end = min(next_start_sec - 0.06, cursor_sec + dur)
            counter_notes.append(
                {
                    "role": "call_response",
                    "section": sec.label,
                    "start_sec": round(cursor_sec, 3),
                    "end_sec": round(n_end, 3),
                    "midi": int(m_note),
                    "note_name": str(librosa.midi_to_note(m_note)),
                    "velocity": 84,
                }
            )
            cursor_sec += eighth_sec

    # 2. High-Fifth / Octave Obligato Pad in Final Chorus(es) on sustained anchors
    for sec in chorus_sections:
        if sec.label not in final_chorus_labels:
            continue
        sec_notes = [(f0_i, f1_i, lm) for (f0_i, f1_i, lm) in note_spans if sec.start_sec <= f0_i * frame_dt < sec.end_sec]
        last_pad_end = -1.0
        for f0_i, f1_i, lead_m in sec_notes:
            t0 = f0_i * frame_dt
            t1 = f1_i * frame_dt
            if (t1 - t0) < 0.55 or t0 < last_pad_end + 0.20:
                continue
            c_vec = np.mean(chord_chroma[:, f0_i:f1_i], axis=1)
            s_vec = np.round(np.mean(local_scale[:, f0_i:f1_i], axis=1))
            fifth_pc = (lead_m + 7) % 12
            fifth_clash = max(float(c_vec[(fifth_pc - 1) % 12]), float(c_vec[(fifth_pc + 1) % 12]))
            use_fifth = (float(c_vec[fifth_pc]) >= 0.45 or float(s_vec[fifth_pc]) > 0.5) and fifth_clash < 0.65
            pad_midi = lead_m + 7 if use_fifth and lead_m + 7 <= 84 else min(84, lead_m + 12)
            counter_notes.append(
                {
                    "role": "obligato_pad",
                    "section": sec.label,
                    "start_sec": round(t0, 3),
                    "end_sec": round(t1, 3),
                    "midi": int(pad_midi),
                    "note_name": str(librosa.midi_to_note(pad_midi)),
                    "velocity": 72,
                }
            )
            last_pad_end = t1

    counter_notes.sort(key=lambda d: (d["start_sec"], d["midi"]))
    return counter_notes


def _add_quantized_midi_track(
    mid: mido.MidiFile,
    track_name: str,
    channel: int,
    program: int,
    notes: list[tuple[float, float, int, int]],
    bpm: float,
    include_tempo: bool = False,
) -> None:
    """Append a cleanly quantized, non-overlapping named MIDI track to `mid`."""
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    if include_tempo:
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(max(bpm, 40.0)), time=0))
    track.append(mido.Message("program_change", channel=channel, program=program, time=0))

    ticks_per_sec = mid.ticks_per_beat * (max(bpm, 40.0) / 60.0)
    grid = max(30, mid.ticks_per_beat // 4)  # 16th-note quantization grid

    events: list[tuple[int, int, mido.Message]] = []
    last_end_tick = 0
    for s_sec, e_sec, pitch, vel in sorted(notes, key=lambda x: (x[0], x[2])):
        t0 = int(round((s_sec * ticks_per_sec) / grid)) * grid
        t1 = int(round((e_sec * ticks_per_sec) / grid)) * grid
        t0 = max(t0, last_end_tick)
        t1 = max(t1, t0 + grid)
        note_val = int(np.clip(pitch, 36, 96))
        vel_val = int(np.clip(vel, 30, 120))
        events.append((t0, 1, mido.Message("note_on", channel=channel, note=note_val, velocity=vel_val, time=0)))
        events.append((t1, 0, mido.Message("note_off", channel=channel, note=note_val, velocity=0, time=0)))
        last_end_tick = t1

    events.sort(key=lambda item: (item[0], item[1]))
    prev_tick = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - prev_tick)
        track.append(msg)
        prev_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))


def _build_harmony_midi_notes(
    note_spans: list[tuple[int, int, int]],
    chord_chroma: np.ndarray,
    local_scale: np.ndarray,
    allowed_intervals: tuple[int, int, int],
    arr_mask: np.ndarray | None,
    sr: int,
    hop: int,
) -> list[tuple[float, float, int, int]]:
    """Compute chord-safe quantized MIDI note tuples `(start_sec, end_sec, midi, velocity)`."""
    frame_dt = hop / float(sr)
    out: list[tuple[float, float, int, int]] = []
    prev_choice = allowed_intervals[0]
    for f0_i, f1_i, lead_m in note_spans:
        if arr_mask is not None:
            s_mid = min(len(arr_mask) - 1, int(((f0_i + f1_i) * 0.5) * hop))
            if float(arr_mask[s_mid]) < 0.05:
                continue
        c_vec = np.mean(chord_chroma[:, f0_i:f1_i], axis=1)
        s_vec = np.round(np.mean(local_scale[:, f0_i:f1_i], axis=1))
        best_int, _, _, _ = _score_chord_safe_interval(
            p_lead=lead_m % 12,
            c_vec=c_vec,
            s_vec=s_vec,
            allowed_intervals=allowed_intervals,
            prev_choice=prev_choice,
        )
        prev_choice = best_int
        out.append((f0_i * frame_dt, f1_i * frame_dt, lead_m + best_int, 82))
    return out


def _write_counter_melody_midi_file(
    out_midi_path: Path,
    bpm: float,
    note_spans: list[tuple[int, int, int]],
    chord_chroma: np.ndarray,
    local_scale: np.ndarray,
    upper_mask: np.ndarray | None,
    lower_mask: np.ndarray | None,
    counter_notes: list[dict],
    sr: int,
    hop: int,
) -> Path:
    """Write the 4-track Director & Harmony MIDI guide cleanly to `out_midi_path`."""
    frame_dt = hop / float(sr)
    lead_notes = [(f0_i * frame_dt, f1_i * frame_dt, lm, 92) for f0_i, f1_i, lm in note_spans]
    upper_notes = _build_harmony_midi_notes(note_spans, chord_chroma, local_scale, (3, 4, 5), upper_mask, sr, hop)
    lower_notes = _build_harmony_midi_notes(note_spans, chord_chroma, local_scale, (-3, -4, -5), lower_mask, sr, hop)
    counter_tuples = [
        (float(d["start_sec"]), float(d["end_sec"]), int(d["midi"]), int(d["velocity"])) for d in counter_notes
    ]
    out_p = Path(out_midi_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    mid = mido.MidiFile(ticks_per_beat=480)
    _add_quantized_midi_track(mid, "Lead Melody Guide", channel=0, program=52, notes=lead_notes, bpm=bpm, include_tempo=True)
    _add_quantized_midi_track(mid, "Upper Harmony (+3rd/4th)", channel=1, program=53, notes=upper_notes, bpm=bpm)
    _add_quantized_midi_track(mid, "Lower Harmony (-3rd/4th)", channel=2, program=53, notes=lower_notes, bpm=bpm)
    _add_quantized_midi_track(
        mid, "Director Counter-Melody & Call-Response", channel=3, program=54, notes=counter_tuples, bpm=bpm
    )
    mid.save(str(out_p))
    return out_p


def generate_counter_melody_and_export_midi(
    out_midi_path: Path,
    bpm: float,
    lead_vocal_mono: np.ndarray,
    inst_mono: np.ndarray,
    sections: list[SongSection],
    sr: int,
) -> tuple[Path, list[dict]]:
    """Propose Call-and-Response + High-5th/Octave obligato counter-melodies and export 4-track MIDI."""
    n = min(len(lead_vocal_mono), len(inst_mono))
    hop = 512
    chord_chroma, local_scale = _compute_sliding_chord_and_scale_profiles(inst_mono[:n], sr=sr, hop=hop)
    f0, voiced_flag, voiced_prob = librosa.pyin(lead_vocal_mono[:n], fmin=70.0, fmax=900.0, sr=sr, hop_length=hop)
    valid_voiced = (~np.isnan(f0)) & voiced_flag & (voiced_prob > 0.65)
    note_spans = _extract_sung_note_spans(f0, valid_voiced)
    upper_mask, lower_mask, _ = _build_section_arrangement_plan(sections, n_samples=n, sr=sr)
    counter_notes = _compute_counter_melody_notes(
        note_spans=note_spans,
        chord_chroma=chord_chroma,
        local_scale=local_scale,
        sections=sections,
        bpm=bpm,
        sr=sr,
        hop=hop,
    )
    saved = _write_counter_melody_midi_file(
        out_midi_path=out_midi_path,
        bpm=bpm,
        note_spans=note_spans,
        chord_chroma=chord_chroma,
        local_scale=local_scale,
        upper_mask=upper_mask,
        lower_mask=lower_mask,
        counter_notes=counter_notes,
        sr=sr,
        hop=hop,
    )
    return saved, counter_notes


def generate_nectar3_harmonies_and_plan(
    lead_vocal_mono: np.ndarray,
    inst_mono: np.ndarray,
    sr: int,
    sections: list[SongSection],
    existing_user_harmonies: list[np.ndarray] | None = None,
    use_vst3: bool = True,
    out_midi_path: Path | None = None,
    bpm: float = 120.0,
) -> HarmonyGenerationResult:
    """Generate Nectar 3 chord-safe Upper & Lower harmonies for both Turn-Key Mix and DAW Guide use."""
    n = len(lead_vocal_mono)
    upper_arr_mask, lower_arr_mask, suggestions = _build_section_arrangement_plan(sections, n_samples=n, sr=sr)

    # Only feed vocal regions where arrangement calls for harmony (saves processing & keeps rests 100% silent!)
    combined_Gate = ((upper_arr_mask > 0.01) | (lower_arr_mask > 0.01)).astype(np.float32)
    gated_lead = (lead_vocal_mono * np.maximum(upper_arr_mask, lower_arr_mask)).astype(np.float32)

    if np.max(np.abs(gated_lead)) < 1e-4:
        zeros_m = np.zeros(n, dtype=np.float32)
        zeros_st = np.zeros((2, n), dtype=np.float32)
        return HarmonyGenerationResult(
            upper_guide_mono=zeros_m,
            lower_guide_mono=zeros_m,
            auto_mix_harmony_stereo=zeros_st,
            engine_used="none_needed",
            total_harmony_notes=0,
            chord_clash_notes_rescued=0,
            suggestions=suggestions,
        )

    hop = 512
    chord_chroma, local_scale = _compute_sliding_chord_and_scale_profiles(inst_mono[:n], sr=sr, hop=hop)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        lead_vocal_mono, fmin=70.0, fmax=900.0, sr=sr, hop_length=hop
    )
    full_valid = (~np.isnan(f0)) & voiced_flag & (voiced_prob > 0.65)
    valid_voiced = full_valid & (combined_Gate[::hop][: len(f0)] > 0.01)

    all_intervals = (3, 4, 5, -3, -4, -5)
    candidates, engine_name = _render_candidate_intervals_via_nectar3(
        gated_lead, sr=sr, intervals=all_intervals, use_vst3=use_vst3
    )

    raw_upper, notes_up, rescued_up = _select_and_stitch_chord_safe_harmony(
        candidates,
        allowed_intervals=(3, 4, 5),
        lead_f0=f0,
        voiced_mask=valid_voiced,
        chord_chroma=chord_chroma,
        local_scale=local_scale,
        sr=sr,
        hop=hop,
    )
    raw_lower, notes_low, rescued_low = _select_and_stitch_chord_safe_harmony(
        candidates,
        allowed_intervals=(-3, -4, -5),
        lead_f0=f0,
        voiced_mask=valid_voiced,
        chord_chroma=chord_chroma,
        local_scale=local_scale,
        sr=sr,
        hop=hop,
    )

    raw_upper *= upper_arr_mask
    raw_lower *= lower_arr_mask

    # Sculpt harmony tone so it sits smoothly behind the lead vocal without consonantal/presence masking
    pocket_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=210.0),
            PeakFilter(cutoff_frequency_hz=2850.0, gain_db=-2.8, q=1.1),
            LowpassFilter(cutoff_frequency_hz=9200.0),
            Compressor(threshold_db=-21.0, ratio=4.0, attack_ms=8.0, release_ms=110.0),
        ]
    )
    sculpted_up = pocket_board(raw_upper[np.newaxis, :].astype(np.float32), sr)[0]
    sculpted_low = pocket_board(raw_lower[np.newaxis, :].astype(np.float32), sr)[0]

    # Normalize guide/DAW stems to healthy -18.5 LUFS active loudness so Studio One faders (-10.5 dB) work cleanly!
    meter = pyln.Meter(sr)

    def _stage_stem(y: np.ndarray, target_lufs: float = -18.5) -> np.ndarray:
        if np.max(np.abs(y)) < 1e-4:
            return y
        lufs = meter.integrated_loudness(y.astype(np.float64))
        if np.isfinite(lufs):
            g = float(np.clip(target_lufs - lufs, -16.0, 18.0))
            y = y * (10.0 ** (g / 20.0))
        pk = float(np.max(np.abs(y)))
        if pk > 0.85:
            y *= 0.85 / pk
        return y.astype(np.float32)

    upper_guide = _stage_stem(sculpted_up, -18.5)
    lower_guide = _stage_stem(sculpted_low, -18.5)

    # Build ducking mask where user's own recorded harmonies are already singing!
    user_duck = np.ones(n, dtype=np.float32)
    if existing_user_harmonies:
        sum_uh = np.zeros(n, dtype=np.float32)
        for uh_st in existing_user_harmonies:
            uh_m = np.mean(uh_st, axis=0) if uh_st.ndim == 2 else uh_st
            c_len = min(n, len(uh_m))
            sum_uh[:c_len] += np.abs(uh_m[:c_len])
        win_u = max(int(0.06 * sr), 512)
        pad_u = np.pad(sum_uh.astype(np.float64) ** 2, (win_u // 2, win_u - win_u // 2 - 1), mode="edge")
        csum_u = np.cumsum(np.insert(pad_u, 0, 0.0))
        uh_env = np.sqrt(np.maximum((csum_u[win_u:] - csum_u[:-win_u]) / win_u, 1e-12))[:n].astype(np.float32)
        uh_pk = float(np.percentile(uh_env, 96))
        if uh_pk > 1e-4:
            uh_active = np.clip(uh_env / (uh_pk * 0.18 + 1e-9), 0.0, 1.0)
            # Where user's recorded harmony is active, duck Nectar 3 by -13 dB (0.22x) so real vocals lead!
            user_duck = 1.0 - 0.78 * uh_active

    # Equal-power stereo placement for turn-key 2mix: Upper at L-45% (0.85 L / 0.52 R), Lower at R+45% (0.52 L / 0.85 R)
    lead_act = lead_vocal_mono[np.abs(lead_vocal_mono) > 1e-4]
    lead_rms = float(np.sqrt(np.mean(lead_act**2) + 1e-12)) if lead_act.size > 0 else 0.08
    up_act = upper_guide[np.abs(upper_guide) > 1e-4]
    up_rms = float(np.sqrt(np.mean(up_act**2) + 1e-12)) if up_act.size > 0 else 0.08
    rel_scale = min((lead_rms * (10.0 ** (-10.8 / 20.0))) / up_rms, 1.2)

    up_mix = upper_guide * rel_scale * user_duck
    low_mix = lower_guide * (rel_scale * 0.88) * user_duck

    auto_mix_stereo = np.stack(
        [
            0.85 * up_mix + 0.52 * low_mix,
            0.52 * up_mix + 0.85 * low_mix,
        ],
        axis=0,
    ).astype(np.float32)

    all_note_spans = _extract_sung_note_spans(f0, full_valid)
    counter_notes = _compute_counter_melody_notes(
        note_spans=all_note_spans,
        chord_chroma=chord_chroma,
        local_scale=local_scale,
        sections=sections,
        bpm=bpm,
        sr=sr,
        hop=hop,
    )
    midi_path_str: str | None = None
    if out_midi_path is not None:
        saved_midi = _write_counter_melody_midi_file(
            out_midi_path=out_midi_path,
            bpm=bpm,
            note_spans=all_note_spans,
            chord_chroma=chord_chroma,
            local_scale=local_scale,
            upper_mask=upper_arr_mask,
            lower_mask=lower_arr_mask,
            counter_notes=counter_notes,
            sr=sr,
            hop=hop,
        )
        midi_path_str = str(saved_midi)

    return HarmonyGenerationResult(
        upper_guide_mono=upper_guide,
        lower_guide_mono=lower_guide,
        auto_mix_harmony_stereo=auto_mix_stereo,
        engine_used=engine_name,
        total_harmony_notes=max(notes_up, notes_low),
        chord_clash_notes_rescued=rescued_up + rescued_low,
        suggestions=suggestions,
        counter_melody_notes=counter_notes,
        midi_guide_path=midi_path_str,
    )
