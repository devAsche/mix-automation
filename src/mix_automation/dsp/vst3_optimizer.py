"""v6 Ultra-Precision Vocal Sculpting, Zero-Sizzle Dynamics, Post-Comp Soothe2 & Dual-Bus Harmony Engine.

Resolves every structural flaw uncovered in v5 listening & stage-by-stage audio forensics:
  1. 16-Band Ultra-Precision `Persistent Q` (replaces 6-band coarse static EQ):
     - Band 1: F0-adaptive 18 dB/oct HPF.
     - Bands 2-11: 10-band ERB/Octave differentiable PyTorch EQ optimizing BOTH 44-song Intrinsic
       Vocal Shape AND song-specific Backing-Complementary Margins (`band_margins_db`) with a
       zero-mean core anchor (impossible to scoop Body/Mud/Mid to fake Air!) + smoothness regularization.
     - Bands 12-13: Automated narrow-Q (`Q = 6.0..7.5`) surgical room/vocal resonance notches.
     - Bands 14-16: 3 real-time Dynamic & Spectral EQ bands inside `Persistent Q` (dynamic 290 Hz
       proximity mud control, ultra-fast 1.2ms dynamic 3.2 kHz harsh-attack tamer, and real-time
       spectral 7.8 kHz HF sizzle silkener) — ALL serialized natively 1:1 into Studio One 5!
  2. Zero-Sizzle, Attack-Taming Serial Dynamics:
     - `Softube VCA` switched from slow transient-boosting `attack=1.75, drive=1.0` to clean
       fast-attack peak control (`attack=0.55..0.72, output_drive=0.1`).
     - `UAD Teletronix LA-2A` optical leveling across ALL archetypes (eliminates `Waves MV2`
       `low_level=16.2`'s `+10.56 dB` quiet-HF noise floor explosion forever!).
     - Midrange-Only Parallel `Softube Saturation Knob` (`HPF 240 Hz` + `LPF 4.0 kHz`!): thickens
       vocal core with ZERO `6–16 kHz` "ジリジリ" HF sizzle!
     - Post-Comp / Post-Sat `Oeksound Soothe2` (4-band surgical calibration targeting `3.1k` bite,
       `6.4k` sibilance, and `10k+` HF sizzle) + 6-Bark-band consonant-needle soft-shaper!
  3. Non-Floating Cohesion (Pre-Unmasked Optimization + Bidirectional Wet-Tail Lock):
     - Applies gentle Bark Mid Unmasking (`0.80..1.45 dB`) BEFORE Stage D CMA-ES optimization so
       vocal loudness & Partial Loudness are NEVER double-stacked!
     - Bidirectionally aligns `wet_tail_ratio_db` and Early Room glue so vocals sit INSIDE the
       backing track's acoustic world.
  4. Separate Main Harmony vs Choir Processing (Inspired by Pro Mix Engineering):
     - Main Harmonies (`+3/-3/harm`): Lead-formant-adaptive Hollow-Core EQ ("声の芯を抜く"),
       ultra-fast attack (`0.8 ms`) + slow release (`340 ms`) `6.5:1` transient-crushing compression
       ("語頭をつぶす"), short pre-delay (`5 ms`) depth-pushing Room/Plate + diffused wide delay,
       leveled richly & audibly at `-8.6 dB` stereo K-LUFS!
     - Choirs (`choir`): Lush wide ensemble halo leveled smoothly at `-11.2 dB` stereo K-LUFS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pyloudnorm as pyln
import torch
from cmaes import CMA
from pedalboard import (
    Compressor,
    Delay,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
    Pedalboard,
    load_plugin,
)
from scipy import ndimage, signal

from mix_automation.analysis.relative_profile import (
    VOCAL_MIX_BANDS,
    SectionRelativeProfile,
    SectionTransferAdjustment,
    _apply_k_weighting_mono,
    classify_backing_archetype,
    compute_section_relative_profile,
)
from mix_automation.config.plugin_registry import get_plugin_map
from mix_automation.dsp.vst3_adapter import (
    PRECISION_EQ_BAND_SPECS,
    configure_clean_plate_r2,
    configure_early_room_phoenix,
    configure_persistent_q_precision,
    configure_slick_eq,
    configure_softube_saturation,
    configure_softube_vca,
    configure_soothe2_precision,
    configure_uad_la2a,
    configure_uad_pultec,
    set_vst3_param_strict,
)


@dataclass(frozen=True)
class VocalChainParams:
    """Exact verified v6 commercial VST3 parameters (16-band PQ, VCA+LA-2A, Mid-Sat, Post-Soothe2, Space)."""

    archetype_chain: str
    selected_plugins: list[str]
    hpf_hz: float
    body_gain_db: float             # Summary: warm_body (260 Hz)
    mud_cut_db: float               # Summary: avg(420, 680 Hz)
    core_mid_gain_db: float         # Summary: avg(1050, 1600 Hz)
    presence_gain_db: float         # Summary: avg(2400, 3600 Hz)
    air_shelf_db: float             # Summary: 10.5 kHz silky shelf
    vca_threshold_db: float         # Verified negative threshold on Softube VCA
    vca_measured_gr_db: float       # Measured active GR from fast-attack VCA bisection
    comp2_param: float              # Calibrated UAD LA-2A peak_reduct (0..100)
    comp2_measured_gr_db: float     # Measured optical body leveling impact
    parallel_sat_amount: float      # Verified Saturation Knob amount (Midrange 240Hz-4kHz ONLY!)
    parallel_sat_wet_pct: float     # Parallel harmonic blend scaled by inst_mid_density_index
    vocal_fader_db: float           # Post-chain trim around staged -18 LUFS
    early_room_send_db: float       # Tight 8m Chamber early reflections for room glue
    slap_delay_send_db: float       # 76/90ms band-limited slapback
    tempo_delay_send_db: float      # Ducked 1/8 ping-pong delay
    plate_reverb_send_db: float     # Chorus-free short studio Plate (R2)
    doubler_send_db: float          # Subtle stereo micro-width
    unmask_mid_dip_db: float        # Gentle pre-calibrated Bark Mid pocket on backing track
    # --- v6 Ultra-Precision Controls (with safe defaults for backwards compatibility) ---
    precision_eq_gains_db: dict[str, float] = field(default_factory=dict)
    surgical_notches: list[list[float]] = field(default_factory=list)
    dynamic_mud_thresh_db: float = -24.0
    dynamic_mud_range_db: float = 3.0
    dynamic_bite_hz: float = 3250.0
    dynamic_bite_thresh_db: float = -23.0
    dynamic_bite_range_db: float = 3.8
    spectral_hf_hz: float = 7800.0
    spectral_hf_thresh_db: float = -28.0
    spectral_hf_range_db: float = 4.2
    vca_attack_norm: float = 0.55
    vca_release_norm: float = 0.65
    vca_output_drive: float = 0.1
    soothe2_depth: float = 3.2
    soothe2_bite_sens: float = 3.8
    soothe2_sib_sens: float = 5.8
    soothe2_hf_sens: float = 4.8
    bark_transient_tame_amt: float = 0.30
    lead_core_formant_hz: float = 1550.0
    la2a_gain_knob: float = 38.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SongSection:
    """Detected structural section with perceptually propagated clip gain."""

    index: int
    label: str
    start_sec: float
    end_sec: float
    is_anchor: bool
    inst_rms_db: float
    clip_gain_db: float
    adjustment: SectionTransferAdjustment | None


def detect_song_sections(
    vocal_mono: np.ndarray,
    inst_mono: np.ndarray,
    sr: int,
    min_section_sec: float = 7.5,
) -> tuple[list[tuple[float, float, str]], int]:
    """Segment song using BOTH upward & downward vocal-competition (400Hz-8.5kHz) arrangement step-changes."""
    duration = min(len(vocal_mono), len(inst_mono)) / sr
    if duration <= min_section_sec * 1.6:
        return [(0.0, duration, "01_Chorus")], 0

    hop = 1024
    freqs, _, Z = signal.stft(inst_mono, fs=sr, nperseg=2048, noverlap=1024)
    mel = np.abs(Z)
    log_mel = np.log1p(100.0 * mel)
    flux = np.sqrt(np.sum(np.diff(log_mel, axis=1, prepend=log_mel[:, :1]) ** 2, axis=0))
    frame_times = np.arange(len(flux)) * (hop / sr)

    comp_m = (freqs >= 400.0) & (freqs <= 8500.0)
    comp_db = 10.0 * np.log10(np.sum((mel[comp_m] ** 2), axis=0) + 1e-12)
    win_step = max(8, int(2.5 * sr / hop))
    comp_smooth = ndimage.uniform_filter1d(comp_db, size=win_step)
    shift = max(6, int(2.2 * sr / hop))
    novelty_drop = np.zeros_like(comp_smooth)
    if len(comp_smooth) > 2 * shift:
        novelty_drop[shift:-shift] = np.abs(comp_smooth[2 * shift :] - comp_smooth[: -2 * shift])
    novelty_drop_norm = novelty_drop / (float(np.percentile(novelty_drop, 95)) + 1e-6)
    flux_norm = flux / (float(np.percentile(flux, 95)) + 1e-6)
    combined_novelty = 0.50 * flux_norm + 0.50 * novelty_drop_norm

    win = max(5, int(1.6 * sr / hop))
    nov_smooth = np.convolve(combined_novelty, np.hanning(win) / np.sum(np.hanning(win)), mode="same")
    min_dist_frames = int(min_section_sec * sr / hop)
    peaks, _ = signal.find_peaks(
        nov_smooth,
        distance=min_dist_frames,
        prominence=float(np.percentile(nov_smooth, 55)) * 0.28,
    )

    boundaries = (
        [0.0]
        + [float(frame_times[p]) for p in peaks if min_section_sec <= frame_times[p] <= duration - min_section_sec]
        + [duration]
    )
    raw_spans = [(round(a, 2), round(b, 2)) for a, b in zip(boundaries[:-1], boundaries[1:]) if b - a >= 5.5]

    sos_hi = signal.butter(3, [1200.0, min(9000.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    inst_midhi = signal.sosfilt(sos_hi, inst_mono)

    chorus_scores: list[float] = []
    vocal_activities: list[float] = []
    for a, b in raw_spans:
        s0, s1 = int(a * sr), int(b * sr)
        i_db = float(20.0 * np.log10(np.sqrt(np.mean(inst_mono[s0:s1] ** 2) + 1e-12)))
        ih_db = float(20.0 * np.log10(np.sqrt(np.mean(inst_midhi[s0:s1] ** 2) + 1e-12)))
        v_db = float(20.0 * np.log10(np.sqrt(np.mean(vocal_mono[s0:s1] ** 2) + 1e-12)))
        chorus_scores.append(0.25 * i_db + 0.75 * ih_db)
        vocal_activities.append(v_db)

    max_v_db = max(vocal_activities)
    active_song_indices = [i for i, v_db in enumerate(vocal_activities) if v_db >= max_v_db - 16.0]
    if not active_song_indices:
        active_song_indices = list(range(len(raw_spans)))

    best_idx = max(active_song_indices, key=lambda idx: chorus_scores[idx] + 0.15 * vocal_activities[idx])
    active_scores = np.array([chorus_scores[i] for i in active_song_indices])
    q_chorus = float(np.percentile(active_scores, 68)) if len(active_scores) >= 3 else chorus_scores[best_idx] - 1.0
    q_quiet = float(np.percentile(active_scores, 30)) if len(active_scores) >= 4 else float(np.min(active_scores)) - 0.5

    labeled: list[tuple[float, float, str]] = []
    for idx, ((a, b), c_sc, v_db) in enumerate(zip(raw_spans, chorus_scores, vocal_activities)):
        if v_db < max_v_db - 16.0:
            role = "Instrumental"
        elif idx == best_idx or c_sc >= q_chorus:
            role = "Chorus"
        elif c_sc <= q_quiet:
            role = "Quiet_Bridge_Verse"
        else:
            role = "Verse_PreChorus"
        labeled.append((a, b, f"{idx + 1:02d}_{role}"))

    return labeled, best_idx


def compute_continuous_distance_envelopes(
    inst_stereo: np.ndarray,
    sr: int,
    anchor_span: tuple[int, int],
    verse_intimacy_shift_db: float = 0.4,
) -> dict[str, np.ndarray]:
    """Track backing 350Hz-7.5kHz competition loudness continuously to keep vocal distance cohesive everywhere."""
    n = inst_stereo.shape[1]
    inst_mid = 0.5 * (inst_stereo[0].astype(np.float64) + inst_stereo[1].astype(np.float64))
    sos_comp = signal.butter(3, [350.0, min(7500.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    comp_sig = signal.sosfilt(sos_comp, inst_mid)

    hop = max(int(sr * 0.25), 512)
    n_f = max(1, int(np.ceil(n / hop)))
    padded = np.pad(comp_sig, (0, n_f * hop - n))
    frm_db = 20.0 * np.log10(np.sqrt(np.mean(padded.reshape(n_f, hop) ** 2, axis=1) + 1e-12))
    frm_smooth_db = ndimage.gaussian_filter1d(frm_db, sigma=2.5)

    a0 = max(0, anchor_span[0] // hop)
    a1 = min(n_f, max(a0 + 2, anchor_span[1] // hop))
    anchor_ref_db = float(np.percentile(frm_smooth_db[a0:a1], 65))

    delta_db = np.clip(frm_smooth_db - anchor_ref_db, -12.0, 3.0)
    drop_mag = np.maximum(-delta_db - 1.2, 0.0)

    intimacy_cushion = float(np.clip(verse_intimacy_shift_db, 0.0, 1.0)) * 0.35
    gain_db_f = np.clip(-drop_mag * 0.36 + np.minimum(drop_mag * 0.12, intimacy_cushion), -2.8, 0.8)
    gain_lin_f = np.power(10.0, gain_db_f / 20.0)
    sat_scale_f = np.clip(1.0 - drop_mag * 0.095, 0.42, 1.0)
    space_scale_f = np.clip(1.0 + drop_mag * 0.055, 0.94, 1.32)
    soften_amt_f = np.clip(drop_mag / 6.0, 0.0, 1.0)

    sample_x = np.arange(n, dtype=np.float64)
    frame_x = np.arange(n_f, dtype=np.float64) * hop + hop * 0.5

    return {
        "gain_env": np.interp(sample_x, frame_x, gain_lin_f).astype(np.float32),
        "sat_env": np.interp(sample_x, frame_x, sat_scale_f).astype(np.float32),
        "space_env": np.interp(sample_x, frame_x, space_scale_f).astype(np.float32),
        "soften_env": np.interp(sample_x, frame_x, soften_amt_f).astype(np.float32),
    }


def apply_dynamic_distance_tone_soften(
    lead_mono: np.ndarray,
    sr: int,
    soften_env: np.ndarray,
    max_cut_db: float = 1.6,
) -> np.ndarray:
    """Gently soften upper-mid edge (2.2-7.2kHz) during quiet backing breakdowns so vocal distance stays cohesive."""
    if float(np.max(soften_env)) < 0.05:
        return lead_mono
    n = min(len(lead_mono), len(soften_env))
    sos = signal.butter(2, [2200.0, min(7200.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    hi_band = signal.sosfiltfilt(sos, lead_mono[:n].astype(np.float64))
    max_atten = 1.0 - (10.0 ** (-max_cut_db / 20.0))
    out = lead_mono.copy()
    out[:n] = (lead_mono[:n].astype(np.float64) - hi_band * (soften_env[:n].astype(np.float64) * max_atten)).astype(
        np.float32
    )
    return out


# --- Stage A: Automated Narrow-Q Resonance Detection & 10-Band Differentiable Backing-Complementary EQ ---


def detect_vocal_surgical_notches(
    vocal_mono: np.ndarray,
    sr: int,
    max_notches: int = 2,
) -> list[list[float]]:
    """Detect up to 2 steady-state narrow-Q (`Q = 6.0..7.5`) room/nasal/mic resonances in `280Hz-5.2kHz`."""
    if len(vocal_mono) < 4096:
        return []
    freqs, _, Z = signal.stft(vocal_mono, fs=sr, nperseg=4096, noverlap=2048)
    P = np.abs(Z) ** 2
    fp = np.sum(P, axis=0)
    act = fp >= (np.percentile(fp, 95) * (10.0 ** (-22.0 / 10.0)))
    if np.sum(act) < 6:
        return []
    psd_db = 10.0 * np.log10(np.mean(P[:, act], axis=1) + 1e-12)

    # Resample onto a uniform 1/24-octave log-frequency grid from 240 Hz to 5800 Hz
    log_grid = np.geomspace(240.0, min(5800.0, sr * 0.42), 180)
    spec_log = np.interp(log_grid, freqs, psd_db)
    # 1/3-octave smoothed baseline (~11 bins) isolates sharp resonances from broad formants
    baseline = ndimage.gaussian_filter1d(spec_log, sigma=7.5)
    prom = spec_log - baseline

    # Restrict candidates strictly to [290 Hz .. 5000 Hz]
    valid = (log_grid >= 290.0) & (log_grid <= 5000.0)
    prom_valid = np.where(valid, prom, 0.0)
    peaks, _ = signal.find_peaks(prom_valid, height=2.9, distance=18)
    if len(peaks) == 0:
        return []

    ranked = sorted(peaks, key=lambda p_i: float(prom_valid[p_i]), reverse=True)
    chosen: list[list[float]] = []
    for pk in ranked:
        f_hz = float(log_grid[pk])
        if any(abs(np.log2(f_hz / prev[0])) < 0.55 for prev in chosen):
            continue
        p_val = float(prom_valid[pk])
        cut_db = round(-float(np.clip(p_val * 0.52, 1.0, 3.2)), 2)
        q_val = 6.5 if f_hz < 1800.0 else 7.2
        chosen.append([round(f_hz, 1), cut_db, q_val])
        if len(chosen) >= max_notches:
            break
    return chosen


def measure_lead_core_formant_hz(vocal_mono: np.ndarray, sr: int) -> float:
    """Measure singer's primary core vocal formant centroid (`1150-2100 Hz`) for Hollow-Core Harmony EQ."""
    if len(vocal_mono) < 2048:
        return 1550.0
    freqs, _, Z = signal.stft(vocal_mono, fs=sr, nperseg=2048, noverlap=1024)
    P = np.abs(Z) ** 2
    fp = np.sum(P, axis=0)
    act = fp >= (np.percentile(fp, 95) * (10.0 ** (-22.0 / 10.0)))
    if np.sum(act) < 4:
        return 1550.0
    psd = np.mean(P[:, act], axis=1)
    m = (freqs >= 850.0) & (freqs <= 2600.0)
    if not np.any(m) or float(np.sum(psd[m])) < 1e-12:
        return 1550.0
    log_c = float(np.sum(np.log2(freqs[m]) * psd[m]) / (np.sum(psd[m]) + 1e-12))
    return round(float(np.clip(2.0**log_c, 1150.0, 2100.0)), 1)


def _rbj_peaking_mag_sq(w: torch.Tensor, f0: float, q: float, gain_db: torch.Tensor, sr: int) -> torch.Tensor:
    A = torch.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    cos_w = torch.cos(w)
    cos_2w = torch.cos(2.0 * w)
    num = b0 * b0 + b1 * b1 + b2 * b2 + 2.0 * (b0 * b1 + b1 * b2) * cos_w + 2.0 * b0 * b2 * cos_2w
    den = a0 * a0 + a1 * a1 + a2 * a2 + 2.0 * (a0 * a1 + a1 * a2) * cos_w + 2.0 * a0 * a2 * cos_2w
    return torch.clamp(num / (den + 1e-12), min=1e-8)


def _rbj_highshelf_mag_sq(w: torch.Tensor, f0: float, q: float, gain_db: torch.Tensor, sr: int) -> torch.Tensor:
    A = torch.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)
    sqA = torch.sqrt(torch.clamp(A, min=1e-4))
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqA * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqA * alpha)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqA * alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqA * alpha
    cos_w = torch.cos(w)
    cos_2w = torch.cos(2.0 * w)
    num = b0 * b0 + b1 * b1 + b2 * b2 + 2.0 * (b0 * b1 + b1 * b2) * cos_w + 2.0 * b0 * b2 * cos_2w
    den = a0 * a0 + a1 * a1 + a2 * a2 + 2.0 * (a0 * a1 + a1 * a2) * cos_w + 2.0 * a0 * a2 * cos_2w
    return torch.clamp(num / (den + 1e-12), min=1e-8)


def solve_precision_10band_eq_gpu(
    vocal_mono: np.ndarray,
    inst_stereo: np.ndarray | None,
    sr: int,
    target_profile: SectionRelativeProfile,
    inst_high_rolloff_db: float,
    median_f0_hz: float = 250.0,
) -> dict[str, object]:
    """Run 260-step PyTorch differentiable gradient descent over 10 ERB/Octave RBJ biquads + Backing Pocket Sync.

    Fixes v5's 6-band scooping bug completely via:
      1. 10 finely-spaced ERB/Octave sculpting points (160, 260, 420, 680, 1050, 1600, 2400, 3600, 5800, 10500 Hz).
      2. Dual objective: matches BOTH 44-song Intrinsic Vocal Shape AND level-invariant Backing-Complementary
         Band Margins (`band_margins_db`) so dark/mellow tracks never get +5 dB air blast!
      3. Zero-Mean Core Anchor (`mean(bands 0..7) ≈ 0`) + adjacent-band smoothness regularizer so the EQ
         cannot cheat by cutting `-4 dB` out of Body/Mud/Mid just to raise relative Air!
    """
    device = torch.device("cpu")
    freqs_np, _, Z_v = signal.stft(vocal_mono, fs=sr, nperseg=2048, noverlap=1024)
    pow_v = np.abs(Z_v) ** 2
    fp = np.sum(pow_v, axis=0)
    act = fp >= (np.percentile(fp, 95) * (10.0 ** (-24.0 / 10.0)))
    if not np.any(act):
        act = np.ones_like(fp, dtype=bool)
    mean_v_psd_np = np.mean(pow_v[:, act], axis=1).astype(np.float32)

    mean_i_psd_t: torch.Tensor | None = None
    if inst_stereo is not None and inst_stereo.size >= 2048:
        inst_m = (
            0.5 * (inst_stereo[0] + inst_stereo[1])
            if inst_stereo.ndim == 2 and inst_stereo.shape[0] >= 2
            else inst_stereo.reshape(-1)
        )
        n_c = min(len(vocal_mono), len(inst_m))
        _, _, Z_i = signal.stft(inst_m[:n_c], fs=sr, nperseg=2048, noverlap=1024)
        pow_i = np.abs(Z_i) ** 2
        n_f = min(pow_i.shape[1], len(act))
        act_i = act[:n_f]
        if not np.any(act_i):
            act_i = np.ones(n_f, dtype=bool)
        mean_i_psd_t = torch.tensor(np.mean(pow_i[:, :n_f][:, act_i], axis=1).astype(np.float32), device=device)

    freqs = torch.tensor(freqs_np, dtype=torch.float32, device=device)
    w = 2.0 * np.pi * freqs / float(sr)
    psd_v = torch.tensor(mean_v_psd_np, dtype=torch.float32, device=device)

    vzone_mask = (freqs >= 180.0) & (freqs <= 6000.0)
    band_masks = {b: (freqs >= lo) & (freqs < hi) for b, (lo, hi) in VOCAL_MIX_BANDS.items()}

    # Smooth HF targets capped gently toward Azusagawa / silky commercial range so home-mic hiss is never blasted
    rolloff_delta = float(np.clip(inst_high_rolloff_db - target_profile.inst_high_rolloff_db, -6.0, 6.0))
    tgt_intrinsic = dict(target_profile.vocal_intrinsic_bands_db)
    tgt_intrinsic["sibilance"] = min(float(tgt_intrinsic["sibilance"]), -18.8)
    tgt_intrinsic["air_sheen"] = min(float(tgt_intrinsic["air_sheen"]) + 0.30 * rolloff_delta, -19.2)

    # Level-invariant backing-relative target margins (relative to core vocal-zone margin!)
    tgt_margins = dict(target_profile.band_margins_db)
    tgt_core_margin = float(
        np.mean([tgt_margins["body_warmth"], tgt_margins["core_mid"], tgt_margins["core_presence"]])
    )
    tgt_rel_margins = {b: float(tgt_margins[b]) - tgt_core_margin for b in VOCAL_MIX_BANDS}

    hpf_hz = round(float(np.clip(median_f0_hz * 0.38, 72.0, 112.0)), 1)
    hpf_mag_sq = 1.0 / (1.0 + torch.pow(hpf_hz / (freqs + 1e-4), 6.0))

    raw_g = torch.zeros(len(PRECISION_EQ_BAND_SPECS), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw_g], lr=0.075)

    intrinsic_weights = {
        "body_warmth": 2.4,
        "boxiness_mud": 2.6,
        "core_mid": 2.5,
        "core_presence": 3.0,
        "sibilance": 2.0,
        "air_sheen": 1.6,
    }
    backing_rel_weights = {
        "body_warmth": 0.45,
        "boxiness_mud": 0.40,
        "core_mid": 0.55,
        "core_presence": 1.15,
        "sibilance": 1.20,
        "air_sheen": 1.35,
    }

    # Per-band smooth (lo, hi) bounds
    band_bounds: list[tuple[float, float]] = [
        (-2.8, +2.6),  # 160 Hz low_chest
        (-2.8, +2.8),  # 260 Hz warm_body
        (-3.6, +1.8),  # 420 Hz low_box
        (-3.6, +1.6),  # 680 Hz mud_honk
        (-2.8, +2.6),  # 1050 Hz throat_core
        (-2.8, +2.6),  # 1600 Hz upper_core
        (-3.0, +2.8),  # 2400 Hz vocal_bite
        (-3.2, +2.6),  # 3600 Hz clarity_pres
        (-3.6, +1.0),  # 5800 Hz sibilant_sheen (guardrail!)
        (-3.6, +1.8),  # 10500 Hz air_shelf (hard silk ceiling!)
    ]

    def _bound_gains(r: torch.Tensor) -> torch.Tensor:
        out = []
        for idx, (lo, hi) in enumerate(band_bounds):
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            out.append(mid + half * torch.tanh(r[idx] / max(half, 0.5)))
        return torch.stack(out)

    i_zone_db_t: torch.Tensor | None = None
    i_band_db_map: dict[str, torch.Tensor] = {}
    if mean_i_psd_t is not None:
        i_zone_db_t = 10.0 * torch.log10(torch.sum(mean_i_psd_t[vzone_mask]) + 1e-12)
        for b_name, msk in band_masks.items():
            i_band_db_map[b_name] = 10.0 * torch.log10(torch.sum(mean_i_psd_t[msk]) + 1e-12)

    for _ in range(260):
        opt.zero_grad()
        g = _bound_gains(raw_g)
        H2 = hpf_mag_sq
        for idx, (_, freq_hz, q_val, b_type) in enumerate(PRECISION_EQ_BAND_SPECS):
            if b_type == "HighShelf":
                H2 = H2 * _rbj_highshelf_mag_sq(w, freq_hz, q_val, g[idx], sr)
            else:
                H2 = H2 * _rbj_peaking_mag_sq(w, freq_hz, q_val, g[idx], sr)

        eq_psd = psd_v * H2
        z_db = 10.0 * torch.log10(torch.sum(eq_psd[vzone_mask]) + 1e-12)

        loss = torch.tensor(0.0, device=device)
        v_band_dbs: dict[str, torch.Tensor] = {}
        z_db_detached = z_db.detach()
        for b_name, wt in intrinsic_weights.items():
            vb_db = 10.0 * torch.log10(torch.sum(eq_psd[band_masks[b_name]]) + 1e-12)
            v_band_dbs[b_name] = vb_db
            # Stop gradient from HF targets (sibilance/air) leaking into z_db and scooping Body/Mud/Mid!
            ref_z = z_db_detached if b_name in ("sibilance", "air_sheen") else z_db
            loss = loss + wt * (((vb_db - ref_z) - float(tgt_intrinsic[b_name])) ** 2)

        # Backing-Complementary Pocket Alignment (level-invariant!)
        if i_zone_db_t is not None:
            cur_core_margin = (
                (v_band_dbs["body_warmth"] - i_band_db_map["body_warmth"])
                + (v_band_dbs["core_mid"] - i_band_db_map["core_mid"])
                + (v_band_dbs["core_presence"] - i_band_db_map["core_presence"])
            ) / 3.0
            cur_core_margin_det = cur_core_margin.detach()
            for b_name, wt_rel in backing_rel_weights.items():
                ref_m = cur_core_margin_det if b_name in ("sibilance", "air_sheen") else cur_core_margin
                rel_m = (v_band_dbs[b_name] - i_band_db_map[b_name]) - ref_m
                diff_m = rel_m - tgt_rel_margins[b_name]
                if b_name in ("core_presence", "sibilance", "air_sheen"):
                    up_extra = torch.clamp(diff_m, min=0.0)
                    loss = loss + wt_rel * (diff_m**2 + 1.2 * (up_extra**2))
                else:
                    loss = loss + wt_rel * (diff_m**2)

        # Gentle Zero-Mean Core Anchor (bands 0..7) + smoothness keeps EQ natural & phase-clean
        core_mean = torch.mean(g[:8])
        loss = loss + 2.2 * (core_mean**2)
        loss = loss + 0.16 * torch.sum((g[1:] - g[:-1]) ** 2)
        loss = loss + 0.09 * torch.sum(g**2)

        loss.backward()
        opt.step()

    with torch.no_grad():
        g_final = _bound_gains(raw_g).cpu().numpy()
        gains_dict = {
            spec[0]: round(float(g_final[idx]), 2) for idx, spec in enumerate(PRECISION_EQ_BAND_SPECS)
        }
        return {
            "hpf_hz": hpf_hz,
            "precision_eq_gains_db": gains_dict,
            "body_gain_db": round(0.5 * (gains_dict["low_chest"] + gains_dict["warm_body"]), 2),
            "mud_cut_db": round(0.5 * (gains_dict["low_box"] + gains_dict["mud_honk"]), 2),
            "core_mid_gain_db": round(0.5 * (gains_dict["throat_core"] + gains_dict["upper_core"]), 2),
            "presence_gain_db": round(0.5 * (gains_dict["vocal_bite"] + gains_dict["clarity_pres"]), 2),
            "air_shelf_db": gains_dict["air_shelf"],
        }


def solve_differentiable_intrinsic_eq_gpu(
    vocal_mono: np.ndarray,
    sr: int,
    target_profile: SectionRelativeProfile,
    inst_high_rolloff_db: float,
    inst_stereo: np.ndarray | None = None,
    median_f0_hz: float = 250.0,
) -> dict[str, float]:
    """Backwards-compatible entry point delegating to v6 10-band differentiable vocal EQ."""
    res = solve_precision_10band_eq_gpu(
        vocal_mono=vocal_mono,
        inst_stereo=inst_stereo,
        sr=sr,
        target_profile=target_profile,
        inst_high_rolloff_db=inst_high_rolloff_db,
        median_f0_hz=median_f0_hz,
    )
    return {
        "hpf_hz": float(res["hpf_hz"]),
        "body_gain_db": float(res["body_gain_db"]),
        "mud_cut_db": float(res["mud_cut_db"]),
        "core_mid_gain_db": float(res["core_mid_gain_db"]),
        "presence_gain_db": float(res["presence_gain_db"]),
        "air_shelf_db": float(res["air_shelf_db"]),
    }


def apply_bark_consonant_transient_tamer(
    mono_sig: np.ndarray,
    sr: int,
    amount: float = 0.35,
    spike_thresh_db: float = 5.2,
    max_tame_db: float = 4.2,
) -> np.ndarray:
    """6-Bark-band (`1.8-6.8 kHz`) short-hop differential-envelope consonant-needle rounder.

    Inspired by `legsmechanical/bark-transient-shaper`: compares a fast ~3ms onset envelope against
    a ~65ms syllabic body envelope per Bark band and gently softens ONLY outlier piercing consonant
    spikes (`> +7 dB` above local vowel body) while leaving vowels, warmth & diction 100% intact!
    """
    if amount <= 0.02 or len(mono_sig) < 2048:
        return mono_sig

    n = len(mono_sig)
    nperseg = 1024
    hop = 128  # ~2.9 ms temporal resolution at 44.1 kHz
    noverlap = nperseg - hop
    freqs, _, Z = signal.stft(mono_sig.astype(np.float64), fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    n_bins, n_frames = Z.shape
    mag = np.abs(Z)

    bark_edges = np.array([1800.0, 2300.0, 2900.0, 3600.0, 4500.0, 5600.0, 6800.0], dtype=np.float64)
    n_bands = len(bark_edges) - 1
    centers = np.sqrt(bark_edges[:-1] * bark_edges[1:])
    band_atten_db = np.zeros((n_bands, n_frames), dtype=np.float64)

    # Fast (~3ms) vs Slow (~65ms) one-pole envelope tracking
    a_fast = 0.65
    a_slow_up = 0.045
    a_slow_dn = 0.035

    for b in range(n_bands):
        m = (freqs >= bark_edges[b]) & (freqs < bark_edges[b + 1])
        if not np.any(m):
            continue
        e = np.sqrt(np.mean(mag[m] ** 2, axis=0) + 1e-14)
        e_floor = float(np.percentile(e, 95)) * (10.0 ** (-26.0 / 20.0))

        fast_e = np.empty(n_frames, dtype=np.float64)
        slow_e = np.empty(n_frames, dtype=np.float64)
        cf = float(e[0])
        cs = float(e[0])
        for t in range(n_frames):
            x = float(e[t])
            cf = cf + a_fast * (x - cf)
            cs = cs + (a_slow_up if x > cs else a_slow_dn) * (x - cs)
            fast_e[t] = cf
            slow_e[t] = cs

        ratio_db = 20.0 * np.log10((fast_e + 1e-12) / (np.maximum(slow_e, e_floor) + 1e-12))
        excess_db = np.maximum(ratio_db - spike_thresh_db, 0.0) * (fast_e > e_floor)
        band_atten_db[b] = np.clip(excess_db * float(np.clip(amount, 0.0, 0.65)), 0.0, max_tame_db)

    bin_atten_db = np.zeros((n_bins, n_frames), dtype=np.float64)
    in_zone = (freqs >= 1500.0) & (freqs <= 7600.0)
    idx_in = np.where(in_zone)[0]
    for bin_i in idx_in:
        f_val = freqs[bin_i]
        if f_val <= centers[0]:
            w_e = np.clip((f_val - 1500.0) / max(centers[0] - 1500.0, 1.0), 0.0, 1.0)
            bin_atten_db[bin_i] = band_atten_db[0] * w_e
        elif f_val >= centers[-1]:
            w_e = np.clip((7600.0 - f_val) / max(7600.0 - centers[-1], 1.0), 0.0, 1.0)
            bin_atten_db[bin_i] = band_atten_db[-1] * w_e
        else:
            hi_b = int(np.searchsorted(centers, f_val))
            lo_b = hi_b - 1
            frac = (f_val - centers[lo_b]) / max(centers[hi_b] - centers[lo_b], 1.0)
            bin_atten_db[bin_i] = (1.0 - frac) * band_atten_db[lo_b] + frac * band_atten_db[hi_b]

    gain_mask = np.power(10.0, -bin_atten_db / 20.0)
    _, y = signal.istft(Z * gain_mask, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)), mode="edge")
    return y.astype(np.float32)


def _measure_atk26_crest_db(sig: np.ndarray, sr: int) -> float:
    """Measure 2-6 kHz consonant attack peak/RMS crest ratio on active frames."""
    if len(sig) < 2048:
        return 13.5
    sos = signal.butter(3, [2000.0, min(6000.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    bp = signal.sosfilt(sos, sig.astype(np.float64))
    nz = np.abs(sig) > (float(np.percentile(np.abs(sig), 98)) * 0.1 + 1e-6)
    if np.sum(nz) < 256:
        return 13.5
    pk = float(np.percentile(np.abs(bp[nz]), 99.7)) + 1e-12
    rm = float(np.sqrt(np.mean(bp[nz] ** 2))) + 1e-12
    return float(20.0 * np.log10(pk / rm))


class RealVST3CharacterRig:
    """v6 Contract-Verified Ultra-Precision Commercial VST3 Rig (16-Band PQ + Fast VCA + LA-2A + Mid-Sat + Post-Soothe2)."""

    def __init__(self, archetype: str, bpm: float) -> None:
        self.archetype = archetype
        self.bpm = bpm
        pmap = get_plugin_map()

        self.pq = load_plugin(str(pmap["persistent_q"].loadable_path))
        self.vca = load_plugin(str(pmap["vca_compressor"].loadable_path))
        self.la2a = load_plugin(str(pmap["uad_la2a"].loadable_path))
        self.sat = load_plugin(str(pmap["saturation_knob"].loadable_path))
        self.soothe2 = load_plugin(str(pmap["soothe2"].loadable_path)) if pmap["soothe2"].exists else None

        self.plugin_names: list[str] = [
            "Persistent Q (16-Band Static+Notch+Dynamic+Spectral EQ)",
            "Softube VCA Compressor (Fast Clean Peak Tamer)",
            "UAD Teletronix LA-2A (Zero-Noise Optical Body Glue)",
        ]

        self.color = None
        if archetype == "intimate_ballad_sparse":
            self.color = load_plugin(str(pmap["slick_eq"].loadable_path))
            self.plugin_names.append("TDR VOS SlickEQ (Silky Analog)")
        elif archetype == "groovy_mid_pop":
            self.color = load_plugin(str(pmap["uad_pultec_eqp1a"].loadable_path))
            self.plugin_names.append("UAD Pultec EQP-1A (Warm Foundation)")

        self.plugin_names.append("Softube Saturation Knob (240Hz-4kHz Midrange-Only Parallel)")
        if self.soothe2 is not None:
            self.plugin_names.append("Oeksound Soothe2 (Post-Comp 4-Band Bite/Sibilance/HF Polish)")

        # 3-Layer Spatial Rig: Early-Reflection Room (PhoenixVerb) + Slap/Ducked 1/8 + Clean Plate (R2)
        self.phoenix = load_plugin(str(pmap["phoenix_verb"].loadable_path)) if pmap["phoenix_verb"].exists else None
        self.r2 = load_plugin(str(pmap["exponential_r2"].loadable_path))
        self._configure_spatial_plugins()
        self.plugin_names.extend(
            ["PhoenixVerb (8m Early Room)", "Slapback 76/90ms + Ducked 1/8 Delay", "Exponential Audio R2 (Clean Plate)"]
        )

    def _configure_spatial_plugins(self) -> None:
        predly_plate = round(float(np.clip(60000.0 / max(self.bpm, 60.0) / 16.0, 18.0, 36.0)), 1)
        r_time = 1.25 if self.archetype == "intimate_ballad_sparse" else 0.94
        configure_clean_plate_r2(self.r2, rvb_time=r_time, predelay_ms=predly_plate, lpf_hz=5400.0)
        if self.phoenix is not None:
            configure_early_room_phoenix(self.phoenix, size_m=8.0, rvb_time=0.44, predelay_ms=10.0)

    def calibrate_dynamics_bisection(
        self,
        eq_vocal_mono: np.ndarray,
        sr: int,
        target_crest_db: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Bisect FAST clean VCA peak-tamer + unity-gain-calibrated LA-2A optical body leveler."""
        st_in = np.stack([eq_vocal_mono, eq_vocal_mono], axis=0).astype(np.float32)
        nz = np.abs(eq_vocal_mono) > 1e-3
        if np.sum(nz) < 1000:
            return -22.0, 2.4, 42.0, 2.5, 0.55, 0.65, 38.0

        in_act_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(eq_vocal_mono[nz] ** 2)) + 1e-12))

        # Fast-attack VCA catches piercing consonant peaks cleanly (0.52-0.72, output_drive=0.1!)
        vca_atk = 0.52 if self.archetype == "dense_rock_edm" else (0.58 if self.archetype == "groovy_mid_pop" else 0.72)
        vca_rel = 0.62 if self.archetype == "dense_rock_edm" else 0.68
        vca_ratio = 3.5 if self.archetype == "dense_rock_edm" else 3.0
        target_vca_gr = 2.8 if self.archetype == "dense_rock_edm" else (2.4 if self.archetype == "groovy_mid_pop" else 1.8)

        lo_t, hi_t = -42.0, -8.0
        best_vca_t = -22.0
        best_vca_gr = 0.0
        best_err = 999.0

        for _ in range(6):
            mid_t = 0.5 * (lo_t + hi_t)
            configure_softube_vca(
                self.vca,
                threshold_db=mid_t,
                ratio=vca_ratio,
                attack_norm=vca_atk,
                release_norm=vca_rel,
                sc_hpf_hz=120.0,
                output_db=0.0,
                dry_wet=100.0,
                output_drive=0.1,
            )
            y_vca = np.mean(self.vca(st_in, sr, reset=True), axis=0)
            out_act_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(y_vca[nz] ** 2)) + 1e-12))
            gr_db = in_act_rms_db - out_act_rms_db
            err = abs(gr_db - target_vca_gr)
            if err < best_err:
                best_err = err
                best_vca_t = mid_t
                best_vca_gr = gr_db
            if gr_db > target_vca_gr:
                lo_t = mid_t
            else:
                hi_t = mid_t

        configure_softube_vca(
            self.vca,
            threshold_db=best_vca_t,
            ratio=vca_ratio,
            attack_norm=vca_atk,
            release_norm=vca_rel,
            sc_hpf_hz=120.0,
            output_db=float(np.clip(best_vca_gr, 0.0, 7.0)),
            dry_wet=100.0,
            output_drive=0.1,
        )
        after_vca = np.mean(self.vca(st_in, sr, reset=True), axis=0)
        st_vca = np.stack([after_vca, after_vca], axis=0).astype(np.float32)
        vca_act_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(after_vca[nz] ** 2)) + 1e-12))

        # 2. Bisect LA-2A `peak_reduct` by MEASURED OPTICAL GR (2.2-3.8 dB) vs `peak_reduct=0` reference!
        configure_uad_la2a(self.la2a, peak_reduct=0.0, gain=35.0)
        ref_la2a = np.mean(self.la2a(st_vca, sr, reset=True), axis=0)
        ref_la2a_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(ref_la2a[nz] ** 2)) + 1e-12))

        target_la2a_gr = 3.6 if self.archetype == "dense_rock_edm" else (2.8 if self.archetype == "groovy_mid_pop" else 2.0)
        lo_p, hi_p = 18.0, 78.0
        best_c2 = 44.0
        best_c2_gr = target_la2a_gr
        best_c_err = 999.0
        for _ in range(5):
            mid_p = 0.5 * (lo_p + hi_p)
            configure_uad_la2a(self.la2a, peak_reduct=mid_p, gain=35.0)
            y2 = np.mean(self.la2a(st_vca, sr, reset=True), axis=0)
            y2_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(y2[nz] ** 2)) + 1e-12))
            gr_opt = max(0.0, ref_la2a_rms_db - y2_rms_db)
            err = abs(gr_opt - target_la2a_gr)
            if err < best_c_err:
                best_c_err = err
                best_c2 = mid_p
                best_c2_gr = gr_opt
            if gr_opt > target_la2a_gr:
                hi_p = mid_p
            else:
                lo_p = mid_p

        # 3. Bisect LA-2A `gain` knob so output active RMS matches `vca_act_rms_db` (exact 0.0 dB unity gain staging!)
        lo_g, hi_g = 24.0, 68.0
        best_la2a_gain = 38.0
        best_g_err = 999.0
        for _ in range(5):
            mid_g = 0.5 * (lo_g + hi_g)
            configure_uad_la2a(self.la2a, peak_reduct=best_c2, gain=mid_g)
            yg = np.mean(self.la2a(st_vca, sr, reset=True), axis=0)
            yg_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(yg[nz] ** 2)) + 1e-12))
            g_err = abs(yg_rms_db - vca_act_rms_db)
            if g_err < best_g_err:
                best_g_err = g_err
                best_la2a_gain = mid_g
            if yg_rms_db > vca_act_rms_db:
                hi_g = mid_g
            else:
                lo_g = mid_g

        return (
            round(best_vca_t, 1),
            round(best_vca_gr, 2),
            round(best_c2, 1),
            round(best_c2_gr, 2),
            vca_atk,
            vca_rel,
            round(best_la2a_gain, 1),
        )

    def apply_insert_chain(
        self,
        vocal_mono: np.ndarray,
        sr: int,
        p: VocalChainParams,
        sat_envelope: np.ndarray | None = None,
        soothe_resolution: str = "high",
    ) -> np.ndarray:
        """Process vocal through v6 16-band PQ -> Fast VCA -> LA-2A -> Mid-Only Sat -> Post-Soothe2 -> Bark Tamer."""
        gains_map = p.precision_eq_gains_db if p.precision_eq_gains_db else {
            "low_chest": p.body_gain_db * 0.8,
            "warm_body": p.body_gain_db,
            "low_box": p.mud_cut_db,
            "mud_honk": p.mud_cut_db * 0.8,
            "throat_core": p.core_mid_gain_db,
            "upper_core": p.core_mid_gain_db * 0.8,
            "vocal_bite": p.presence_gain_db,
            "clarity_pres": p.presence_gain_db * 0.8,
            "sibilant_sheen": 0.0,
            "air_shelf": p.air_shelf_db,
        }
        notches_t = [(float(x[0]), float(x[1]), float(x[2])) for x in p.surgical_notches]

        configure_persistent_q_precision(
            self.pq,
            hpf_hz=p.hpf_hz,
            band_gains_db=gains_map,
            surgical_notches=notches_t,
            dynamic_mud_thresh_db=p.dynamic_mud_thresh_db,
            dynamic_mud_range_db=p.dynamic_mud_range_db,
            dynamic_bite_hz=p.dynamic_bite_hz,
            dynamic_bite_thresh_db=p.dynamic_bite_thresh_db,
            dynamic_bite_range_db=p.dynamic_bite_range_db,
            spectral_hf_hz=p.spectral_hf_hz,
            spectral_hf_thresh_db=p.spectral_hf_thresh_db,
            spectral_hf_range_db=p.spectral_hf_range_db,
            enable_dynamic_bands=True,
        )
        vca_ratio = 3.5 if self.archetype == "dense_rock_edm" else 3.0
        configure_softube_vca(
            self.vca,
            threshold_db=p.vca_threshold_db,
            ratio=vca_ratio,
            attack_norm=p.vca_attack_norm,
            release_norm=p.vca_release_norm,
            sc_hpf_hz=120.0,
            output_db=float(np.clip(p.vca_measured_gr_db, 0.0, 7.0)),
            dry_wet=100.0,
            output_drive=p.vca_output_drive,
        )
        configure_uad_la2a(self.la2a, peak_reduct=p.comp2_param, gain=p.la2a_gain_knob)

        chain = [self.pq, self.vca, self.la2a]
        if self.color is not None:
            if "hf_boost" in self.color.parameters:
                # Warm foundation only — ZERO extra 12kHz sizzle boost stacked onto air!
                configure_uad_pultec(self.color, lf_boost=0.45, hf_boost=0.0, hf_freq="12 KCS")
            elif "out_stage" in self.color.parameters:
                configure_slick_eq(self.color, out_stage="Silky", drive_db=3.2)
            chain.append(self.color)

        board = Pedalboard(chain)
        st_in = np.stack([vocal_mono, vocal_mono], axis=0).astype(np.float32)
        core_st = board(st_in, sr, reset=True)
        core_mono = np.mean(core_st, axis=0, dtype=np.float32)

        # Midrange-ONLY Parallel Harmonic Saturation (240 Hz - 4.0 kHz strictly!! Zero HF sizzle!)
        if p.parallel_sat_amount > 0.1 and p.parallel_sat_wet_pct > 0.01:
            configure_softube_saturation(self.sat, amount=p.parallel_sat_amount, sat_type="KEEP LOW")
            # Band-limit BOTH before AND after saturation so upper harmonics above 4.8kHz NEVER fizz!
            sat_pre = Pedalboard(
                [HighpassFilter(240.0), LowpassFilter(4000.0), self.sat, LowpassFilter(4800.0)]
            )(np.stack([core_mono, core_mono], axis=0), sr, reset=True)
            sat_m = np.mean(sat_pre, axis=0, dtype=np.float32)
            c_rms = float(np.sqrt(np.mean(core_mono**2) + 1e-12))
            s_rms = float(np.sqrt(np.mean(sat_m**2) + 1e-12))
            if s_rms > 1e-6:
                sat_m *= c_rms / s_rms
            base_w = float(np.clip(p.parallel_sat_wet_pct / 100.0, 0.0, 0.22))
            if sat_envelope is not None and len(sat_envelope) >= len(core_mono):
                w_arr = np.clip(base_w * sat_envelope[: len(core_mono)].astype(np.float32), 0.0, 0.24)
                core_mono = ((1.0 - w_arr) * core_mono + w_arr * sat_m).astype(np.float32)
            else:
                core_mono = ((1.0 - base_w) * core_mono + base_w * sat_m).astype(np.float32)

        # Post-Comp / Post-Sat 4-Band Surgical `Oeksound Soothe2` (eliminates harsh bite & HF sizzle!)
        if self.soothe2 is not None and p.soothe2_depth > 0.15:
            configure_soothe2_precision(
                self.soothe2,
                depth=p.soothe2_depth,
                bite_sens_db=p.soothe2_bite_sens,
                sibilance_sens_db=p.soothe2_sib_sens,
                hf_sizzle_sens_db=p.soothe2_hf_sens,
                resolution=soothe_resolution,
            )
            s2_out = self.soothe2(np.stack([core_mono, core_mono], axis=0).astype(np.float32), sr, reset=True)
            core_mono = np.mean(s2_out, axis=0, dtype=np.float32)

        # Gentle 6-Bark-band consonant-needle soft-shaper (rounds extreme piercing onset spikes only!)
        if p.bark_transient_tame_amt > 0.03:
            core_mono = apply_bark_consonant_transient_tamer(
                core_mono, sr=sr, amount=p.bark_transient_tame_amt
            )

        return (core_mono * float(10.0 ** (p.vocal_fader_db / 20.0))).astype(np.float32)

    def render_spatial_blend(
        self,
        dry_lead_mono: np.ndarray,
        sr: int,
        p: VocalChainParams,
        section_space_env: np.ndarray | None = None,
        section_delay_env: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render Early-Reflection Room Glue + Slapback + Ducked 1/8 Delay + Chorus-Free Plate Reverb."""
        st_in = np.stack([dry_lead_mono, dry_lead_mono], axis=0).astype(np.float32)

        win = max(int(sr * 0.05), 512)
        pad = np.pad(dry_lead_mono.astype(np.float64) ** 2, (win // 2, win - win // 2 - 1), mode="edge")
        csum = np.cumsum(np.insert(pad, 0, 0.0))
        env = np.sqrt(np.maximum((csum[win:] - csum[:-win]) / win, 1e-12))[: len(dry_lead_mono)].astype(np.float32)
        act = np.clip(env / (float(np.percentile(env, 92)) + 1e-9), 0.0, 1.0)
        duck_gain = 1.0 - 0.48 * act

        sp_env = (
            section_space_env[: len(dry_lead_mono)].astype(np.float32)[np.newaxis, :]
            if section_space_env is not None and len(section_space_env) >= len(dry_lead_mono)
            else 1.0
        )
        dl_env = (
            section_delay_env[: len(dry_lead_mono)].astype(np.float32)[np.newaxis, :]
            if section_delay_env is not None and len(section_delay_env) >= len(dry_lead_mono)
            else 1.0
        )

        # 1. Early Reflection Room Glue (puts dry vocal INSIDE the backing track's acoustic room!)
        if self.phoenix is not None:
            er_pre = Pedalboard([HighpassFilter(320.0), LowpassFilter(5400.0)])(st_in, sr)
            er_st = self.phoenix(er_pre, sr, reset=True) * (10.0 ** (p.early_room_send_db / 20.0)) * sp_env
        else:
            er_st = np.zeros_like(st_in)

        # 2. Tight Band-Limited Stereo Slapback Delay (76ms L / 90ms R)
        slap_l = Pedalboard([HighpassFilter(420.0), LowpassFilter(3600.0), Delay(0.076, feedback=0.08, mix=1.0)])(
            dry_lead_mono[np.newaxis, :], sr
        )[0]
        slap_r = Pedalboard([HighpassFilter(420.0), LowpassFilter(3600.0), Delay(0.090, feedback=0.08, mix=1.0)])(
            dry_lead_mono[np.newaxis, :], sr
        )[0]
        slap_st = np.stack([slap_l, slap_r], axis=0) * (10.0 ** (p.slap_delay_send_db / 20.0)) * dl_env

        # 3. Tempo-Synced 1/8 Note Filtered Stereo Delay (Ducked + Section Automated)
        eighth_sec = float(np.clip(60.0 / max(self.bpm, 60.0) * 0.5, 0.16, 0.42))
        dly_board = Pedalboard(
            [HighpassFilter(450.0), LowpassFilter(3800.0), Delay(eighth_sec, feedback=0.20, mix=1.0)]
        )
        dly_st = dly_board(st_in, sr) * duck_gain[np.newaxis, :] * (10.0 ** (p.tempo_delay_send_db / 20.0)) * dl_env

        # 4. Clean Tight Plate Reverb (R2, zero chorus, ducked + Section Automated)
        rev_pre = Pedalboard([HighpassFilter(440.0), LowpassFilter(5200.0)])(st_in, sr)
        rev_st = (
            self.r2(rev_pre, sr, reset=True)
            * (0.68 + 0.32 * duck_gain[np.newaxis, :])
            * (10.0 ** (p.plate_reverb_send_db / 20.0))
            * sp_env
        )

        return (er_st + slap_st + dly_st + rev_st).astype(np.float32)


def _soften_harmony_sibilants(proc: np.ndarray, sr: int, cutoff_hz: float = 4600.0, max_depth: float = 0.54) -> np.ndarray:
    """Dynamically melt harsh harmony 's/t/k/ch' fricative bursts so consonants never compete with lead."""
    n = proc.shape[1]
    sos_sib = signal.butter(3, cutoff_hz, btype="highpass", fs=sr, output="sos")
    win = max(int(sr * 0.022), 256)
    out = proc.copy()
    for ch in range(2):
        sib_b = signal.sosfiltfilt(sos_sib, out[ch].astype(np.float64))
        pad_s = np.pad(sib_b**2, (win // 2, win - win // 2 - 1), mode="edge")
        cs = np.cumsum(np.insert(pad_s, 0, 0.0))
        sib_env = np.sqrt(np.maximum((cs[win:] - cs[:-win]) / win, 1e-12))[:n]
        sib_ref = float(np.percentile(sib_env[sib_env > 1e-5], 86)) if np.any(sib_env > 1e-5) else 1.0
        excess = np.clip((sib_env / (sib_ref + 1e-9) - 0.48) / 0.80, 0.0, 1.0)
        out[ch] = (out[ch].astype(np.float64) - sib_b * (excess * max_depth)).astype(np.float32)
    return out


def _level_stereo_bus_to_lead_klufs(
    wide: np.ndarray,
    lead_final_dry: np.ndarray,
    sr: int,
    target_rel_db: float,
) -> np.ndarray:
    """Match a stereo harmony/choir bus to exact TRUE stereo K-weighted active loudness relative to lead."""
    n = min(wide.shape[1], len(lead_final_dry))
    lk = _apply_k_weighting_mono(lead_final_dry[:n].astype(np.float64), sr)
    hk_l = _apply_k_weighting_mono(wide[0, :n].astype(np.float64), sr)
    hk_r = _apply_k_weighting_mono(wide[1, :n].astype(np.float64), sr)

    l_mask = np.abs(lead_final_dry[:n]) > 1e-3
    h_mask = (np.abs(wide[0, :n]) + np.abs(wide[1, :n])) > 2e-4
    if np.sum(l_mask) > 512 and np.sum(h_mask) > 512:
        l_k_rms = float(np.sqrt(np.mean(lk[l_mask] ** 2) + 1e-12))
        h_k_rms = float(np.sqrt(0.5 * (np.mean(hk_l[h_mask] ** 2) + np.mean(hk_r[h_mask] ** 2)) + 1e-12))
        target_rms = l_k_rms * float(10.0 ** (target_rel_db / 20.0))
        gain = float(np.clip(target_rms / h_k_rms, 0.04, 16.0))
        wide = (wide * gain).astype(np.float32)

    out = np.zeros((2, len(lead_final_dry)), dtype=np.float32)
    out[:, :n] = wide[:, :n]
    return out


def process_main_harmony_bus(
    harm_stereo: np.ndarray,
    lead_final_dry: np.ndarray,
    sr: int,
    lead_core_hz: float = 1550.0,
    target_rel_db: float = -8.6,
    side_boost_db: float = 1.6,
    plate_plugin=None,
    room_plugin=None,
) -> np.ndarray:
    """Pro Hollow-Core + Transient-Crushed + Depth-Pushed Main Harmony Processor (`+3/-3/harm`).

    Inspired by pro Japanese Utattemita harmony engineering:
      1. Lead-Formant-Adaptive Hollow-Core EQ (`-4.4 dB @ lead_core_hz`, `-3.8 dB @ 2.85 kHz`,
         warm chord richness `+1.0 dB @ 420 Hz`, `HPF 240 Hz`, `LPF 8.5 kHz`) removes vocal "core"
         so harmonies support the chords richly without competing with the lead's center presence.
      2. Ultra-Fast Attack (`0.8 ms`) + Slow Release (`340 ms`) `6.5:1` Transient-Crushing Compression
         ("語頭をつぶす") flattens syllable onset edges so harmonies glide behind the lead smoothly.
      3. Dynamic Sibilance Melt (`4.6 kHz+`) + Short-Predelay (`5 ms`) Depth-Pushing Room/Plate +
         Diffused Stereo Micro-Delay (`19ms L / 29ms R`) pushes harmonies back on the Z-axis.
      4. Because core/onset/depth separation is clean, Main Harmonies sit audibly & richly at `-8.6 dB`!
    """
    n = min(harm_stereo.shape[1], len(lead_final_dry))
    h = harm_stereo[:, :n].astype(np.float32)
    if np.sum(np.abs(h) > 1e-4) < 1000:
        return np.zeros((2, len(lead_final_dry)), dtype=np.float32)

    core_f = float(np.clip(lead_core_hz, 1100.0, 2150.0))

    # 1. Hollow-Core EQ + Ultra-Fast-Attack / Slow-Release Onset-Crushing Compression
    h_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=240.0),
            PeakFilter(cutoff_frequency_hz=420.0, gain_db=+1.0, q=1.1),
            PeakFilter(cutoff_frequency_hz=core_f, gain_db=-4.4, q=1.05),
            PeakFilter(cutoff_frequency_hz=2850.0, gain_db=-3.8, q=1.10),
            LowpassFilter(cutoff_frequency_hz=8500.0),
            Compressor(threshold_db=-27.0, ratio=6.5, attack_ms=0.8, release_ms=340.0),
        ]
    )
    proc = _soften_harmony_sibilants(h_board(h, sr), sr=sr, cutoff_hz=4600.0, max_depth=0.56)

    # 2. Controlled Stereo Frame Width
    mid = 0.5 * (proc[0] + proc[1])
    side = 0.5 * (proc[0] - proc[1]) * float(10.0 ** (side_boost_db / 20.0))
    wide = np.stack([mid + side, mid - side], axis=0).astype(np.float32)

    # 3. Diffused Wide Stereo Micro-Delay (19ms L / 29ms R) adds lush horizontal glue behind the lead
    dly_l = Pedalboard([HighpassFilter(460.0), LowpassFilter(4200.0), Delay(0.019, feedback=0.10, mix=1.0)])(
        wide[0:1], sr
    )[0]
    dly_r = Pedalboard([HighpassFilter(460.0), LowpassFilter(4200.0), Delay(0.029, feedback=0.10, mix=1.0)])(
        wide[1:2], sr
    )[0]
    wide = (wide + np.stack([dly_l, dly_r], axis=0) * 0.17).astype(np.float32)

    # 4. Short-Predelay (5ms!) Room + Plate pushes Main Harmonies physically BEHIND the lead on Z-axis!
    if room_plugin is not None:
        try:
            orig_pd = getattr(room_plugin, "predelay", None)
            configure_early_room_phoenix(room_plugin, size_m=8.0, rvb_time=0.46, predelay_ms=5.0)
            er = room_plugin(Pedalboard([HighpassFilter(320.0), LowpassFilter(5000.0)])(wide, sr), sr, reset=True)
            wide = (wide + er * 0.22).astype(np.float32)
            if orig_pd is not None:
                set_vst3_param_strict(room_plugin, "predelay", orig_pd)
        except Exception:
            pass
    if plate_plugin is not None:
        try:
            orig_pd2 = getattr(plate_plugin, "predelay", None)
            set_vst3_param_strict(plate_plugin, "predelay", 6.0)
            wet = plate_plugin(Pedalboard([HighpassFilter(420.0), LowpassFilter(4800.0)])(wide, sr), sr, reset=True)
            wide = (wide + wet * 0.26).astype(np.float32)
            if orig_pd2 is not None:
                set_vst3_param_strict(plate_plugin, "predelay", orig_pd2)
        except Exception:
            pass

    return _level_stereo_bus_to_lead_klufs(wide, lead_final_dry, sr=sr, target_rel_db=target_rel_db)


def process_choir_bus(
    choir_stereo: np.ndarray,
    lead_final_dry: np.ndarray,
    sr: int,
    target_rel_db: float = -11.2,
    side_boost_db: float = 1.8,
    plate_plugin=None,
    room_plugin=None,
) -> np.ndarray:
    """Lush Wide Ensemble Choir Halo Processor (`choir1..N`), separated cleanly from Main Harmonies."""
    n = min(choir_stereo.shape[1], len(lead_final_dry))
    h = choir_stereo[:, :n].astype(np.float32)
    if np.sum(np.abs(h) > 1e-4) < 1000:
        return np.zeros((2, len(lead_final_dry)), dtype=np.float32)

    c_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=260.0),
            PeakFilter(cutoff_frequency_hz=520.0, gain_db=-2.5, q=1.15),
            PeakFilter(cutoff_frequency_hz=2550.0, gain_db=-4.5, q=1.0),
            LowpassFilter(cutoff_frequency_hz=7200.0),
            Compressor(threshold_db=-25.5, ratio=5.5, attack_ms=2.2, release_ms=220.0),
        ]
    )
    proc = _soften_harmony_sibilants(c_board(h, sr), sr=sr, cutoff_hz=4500.0, max_depth=0.58)

    mid = 0.5 * (proc[0] + proc[1])
    side = 0.5 * (proc[0] - proc[1]) * float(10.0 ** (side_boost_db / 20.0))
    wide = np.stack([mid + side, mid - side], axis=0).astype(np.float32)

    if room_plugin is not None:
        try:
            er = room_plugin(Pedalboard([HighpassFilter(340.0), LowpassFilter(4800.0)])(wide, sr), sr, reset=True)
            wide = (wide + er * 0.24).astype(np.float32)
        except Exception:
            pass
    if plate_plugin is not None:
        try:
            wet = plate_plugin(Pedalboard([HighpassFilter(440.0), LowpassFilter(4600.0)])(wide, sr), sr, reset=True)
            wide = (wide + wet * 0.28).astype(np.float32)
        except Exception:
            pass

    return _level_stereo_bus_to_lead_klufs(wide, lead_final_dry, sr=sr, target_rel_db=target_rel_db)


def process_dedicated_harmony_bus(
    harm_stereo: np.ndarray,
    lead_final_dry: np.ndarray,
    sr: int,
    target_rel_db: float = -8.8,
    side_boost_db: float = 1.6,
    plate_plugin=None,
    room_plugin=None,
    lead_core_hz: float = 1550.0,
) -> np.ndarray:
    """Backwards-compatible entry point delegating to v6 Hollow-Core Transient-Crushed Harmony Processor."""
    return process_main_harmony_bus(
        harm_stereo=harm_stereo,
        lead_final_dry=lead_final_dry,
        sr=sr,
        lead_core_hz=lead_core_hz,
        target_rel_db=target_rel_db,
        side_boost_db=side_boost_db,
        plate_plugin=plate_plugin,
        room_plugin=room_plugin,
    )


def apply_2mix_bus_glue_and_safe_master(
    inst_stereo: np.ndarray,
    vocal_bus_stereo: np.ndarray,
    sr: int,
    target_lufs: float = -9.0,
) -> np.ndarray:
    """Verified cohesive 2-mix Softube VCA bus glue (~1.3 dB GR, clean drive=0.2) + true-peak mastering."""
    pmap = get_plugin_map()
    meter = pyln.Meter(sr)
    orig_inst_lufs = meter.integrated_loudness(inst_stereo.T.astype(np.float64))
    effective_target_lufs = (
        float(np.clip(orig_inst_lufs + 2.6, -12.0, target_lufs)) if np.isfinite(orig_inst_lufs) else target_lufs
    )

    raw_2mix = (inst_stereo + vocal_bus_stereo).astype(np.float32)

    pre_peak = float(np.max(np.abs(raw_2mix))) + 1e-9
    safe_ceiling = 10.0 ** (-3.5 / 20.0)
    if pre_peak > safe_ceiling:
        raw_2mix *= safe_ceiling / pre_peak

    glued = raw_2mix
    if pmap["vca_compressor"].exists:
        try:
            bus_comp = load_plugin(str(pmap["vca_compressor"].loadable_path))
            mix_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(raw_2mix**2)) + 1e-12))
            configure_softube_vca(
                bus_comp,
                threshold_db=float(np.clip(mix_rms_db - 4.5, -36.0, -10.0)),
                ratio=1.6,
                attack_norm=2.10,
                release_norm=0.70,
                sc_hpf_hz=110.0,
                output_db=1.0,
                dry_wet=100.0,
                output_drive=0.2,
            )
            glued = bus_comp(raw_2mix, sr, reset=True)
        except Exception:
            glued = raw_2mix

    cur_lufs = meter.integrated_loudness(glued.T.astype(np.float64))
    gain_db = float(np.clip(effective_target_lufs - cur_lufs, -6.0, 14.0)) if np.isfinite(cur_lufs) else 0.0
    bus_scaled = (glued * (10.0 ** (gain_db / 20.0))).astype(np.float32)

    if pmap["wave_breaker"].exists:
        try:
            wb = load_plugin(str(pmap["wave_breaker"].loadable_path))
            set_vst3_param_strict(wb, "threshold_db", -0.8)
            return np.clip(wb(bus_scaled, sr, reset=True), -0.985, 0.985).astype(np.float32)
        except Exception:
            pass

    peak_f = float(np.max(np.abs(bus_scaled))) + 1e-9
    if peak_f > 0.98:
        bus_scaled *= 0.98 / peak_f
    return bus_scaled.astype(np.float32)


def apply_dynamic_instrumental_unmasking(
    inst_stereo: np.ndarray,
    vocal_mono: np.ndarray,
    sr: int,
    max_dip_db: float = 1.4,
) -> np.ndarray:
    """Gentle 14-Band Bark adaptive Mid-channel spectral unmasking (300Hz-5.8kHz, Side 100% intact!)."""
    if max_dip_db <= 0.05:
        return inst_stereo

    n = min(inst_stereo.shape[1], len(vocal_mono))
    if n < 2048:
        return inst_stereo

    mid = 0.5 * (inst_stereo[0, :n].astype(np.float64) + inst_stereo[1, :n].astype(np.float64))
    side = 0.5 * (inst_stereo[0, :n].astype(np.float64) - inst_stereo[1, :n].astype(np.float64))
    voc = vocal_mono[:n].astype(np.float64)

    nperseg = 2048
    noverlap = 1536
    freqs, _, Z_mid = signal.stft(mid, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, _, Z_voc = signal.stft(voc, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)

    n_bins, n_frames = Z_mid.shape
    P_mid = np.abs(Z_mid) ** 2
    P_voc = np.abs(Z_voc) ** 2

    band_edges = np.array(
        [300, 420, 560, 720, 920, 1160, 1450, 1800, 2200, 2700, 3300, 4000, 4800, 5800],
        dtype=np.float64,
    )
    n_bands = len(band_edges) - 1
    band_centers = np.sqrt(band_edges[:-1] * band_edges[1:])
    band_dips_db = np.zeros((n_bands, n_frames), dtype=np.float64)

    band_importance = np.array(
        [0.45, 0.55, 0.70, 0.82, 0.95, 1.00, 1.00, 1.00, 1.00, 0.95, 0.85, 0.65, 0.45],
        dtype=np.float64,
    )

    v_total = np.sqrt(np.sum(P_voc[(freqs >= 250) & (freqs <= 6000)], axis=0) + 1e-16)
    v_ref_95 = float(np.percentile(v_total, 95)) + 1e-12
    global_gate = np.clip((v_total / v_ref_95 - 0.06) / 0.28, 0.0, 1.0)

    alpha_atk = 0.38
    alpha_rel = 0.10

    for b in range(n_bands):
        m = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        if not np.any(m):
            continue
        v_b = np.sqrt(np.mean(P_voc[m], axis=0) + 1e-16)
        i_b = np.sqrt(np.mean(P_mid[m], axis=0) + 1e-16)

        v_b_norm = np.clip(v_b / (float(np.percentile(v_b, 95)) + 1e-12), 0.0, 1.0)
        i_b_norm = np.clip(i_b / (float(np.percentile(i_b, 85)) + 1e-12), 0.25, 1.0)
        raw_dip = max_dip_db * band_importance[b] * v_b_norm * i_b_norm * global_gate

        env = np.empty(n_frames, dtype=np.float64)
        cur = 0.0
        for t in range(n_frames):
            x = raw_dip[t]
            cur = cur + (alpha_atk if x > cur else alpha_rel) * (x - cur)
            env[t] = cur
        band_dips_db[b] = env

    bin_dip_db = np.zeros((n_bins, n_frames), dtype=np.float64)
    in_zone = (freqs >= 220.0) & (freqs <= 6800.0)
    idx_in = np.where(in_zone)[0]
    for b_idx, bin_i in enumerate(idx_in):
        f_val = freqs[bin_i]
        if f_val <= band_centers[0]:
            w_edge = np.clip((f_val - 220.0) / max(band_centers[0] - 220.0, 1.0), 0.0, 1.0)
            bin_dip_db[bin_i] = band_dips_db[0] * w_edge
        elif f_val >= band_centers[-1]:
            w_edge = np.clip((6800.0 - f_val) / max(6800.0 - band_centers[-1], 1.0), 0.0, 1.0)
            bin_dip_db[bin_i] = band_dips_db[-1] * w_edge
        else:
            hi_b = int(np.searchsorted(band_centers, f_val))
            lo_b = hi_b - 1
            frac = (f_val - band_centers[lo_b]) / max(band_centers[hi_b] - band_centers[lo_b], 1.0)
            bin_dip_db[bin_i] = (1.0 - frac) * band_dips_db[lo_b] + frac * band_dips_db[hi_b]

    gain_mask = np.power(10.0, -np.clip(bin_dip_db, 0.0, max_dip_db + 0.3) / 20.0)
    _, mid_ducked = signal.istft(Z_mid * gain_mask, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    mid_ducked = mid_ducked[:n]
    if len(mid_ducked) < n:
        mid_ducked = np.pad(mid_ducked, (0, n - len(mid_ducked)), mode="edge")

    out = inst_stereo.copy()
    out[0, :n] = (mid_ducked + side).astype(np.float32)
    out[1, :n] = (mid_ducked - side).astype(np.float32)
    return out.astype(np.float32)


def optimize_real_vst3_chain(
    anchor_vocal_mono: np.ndarray,
    anchor_inst_stereo: np.ndarray,
    sr: int,
    bpm: float,
    target_profile: SectionRelativeProfile,
    n_iterations: int = 36,
    seed: int = 42,
    forced_archetype: str | None = None,
    median_f0_hz: float = 250.0,
) -> tuple[RealVST3CharacterRig, VocalChainParams, SectionRelativeProfile, float]:
    """v6 Hybrid Ultra-Precision Optimizer: 10-Band Diff EQ + Notches + Dynamic/Spectral PQ -> Fast VCA + LA-2A -> Pre-Unmasked 6D CMA-ES."""
    init_prof = compute_section_relative_profile(anchor_vocal_mono, anchor_inst_stereo, sr=sr)
    archetype = forced_archetype or classify_backing_archetype(init_prof.inst_crest_db, init_prof.inst_band_levels_db)

    rig = RealVST3CharacterRig(archetype=archetype, bpm=bpm)

    eval_len = min(len(anchor_vocal_mono), sr * 12)
    v_eval = anchor_vocal_mono[:eval_len]
    i_eval_st = anchor_inst_stereo[:, :eval_len]

    # --- Stage A: 10-Band Differentiable Backing-Complementary EQ + Auto Narrow-Q Resonance Notches ---
    eq_init = solve_precision_10band_eq_gpu(
        v_eval,
        inst_stereo=i_eval_st,
        sr=sr,
        target_profile=target_profile,
        inst_high_rolloff_db=init_prof.inst_high_rolloff_db,
        median_f0_hz=median_f0_hz,
    )
    base_gains: dict[str, float] = dict(eq_init["precision_eq_gains_db"])  # type: ignore[arg-type]
    notches = detect_vocal_surgical_notches(anchor_vocal_mono, sr=sr, max_notches=2)
    lead_core_hz = measure_lead_core_formant_hz(anchor_vocal_mono, sr=sr)

    # Calibrate Persistent Q Dynamic/Spectral thresholds relative to staged anchor vocal active RMS
    nz = np.abs(v_eval) > 1e-3
    v_act_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(v_eval[nz] ** 2)) + 1e-12)) if np.any(nz) else -18.0
    dyn_mud_th = round(float(np.clip(v_act_rms_db - 6.5, -36.0, -14.0)), 1)
    dyn_bite_th = round(float(np.clip(v_act_rms_db - 5.5, -34.0, -14.0)), 1)
    spec_hf_th = round(float(np.clip(v_act_rms_db - 11.5, -42.0, -18.0)), 1)

    # Measure raw 2-6kHz consonant attack spikiness across the anchor vocal to adapt dynamic bite & Bark tamer!
    raw_atk26 = _measure_atk26_crest_db(anchor_vocal_mono, sr=sr)
    bite_excess = max(0.0, raw_atk26 - 13.4)
    dyn_bite_range = round(float(np.clip(1.8 + bite_excess * 1.4, 1.8, 4.2)), 2)
    s2_bite_sens = round(float(np.clip(1.4 + bite_excess * 1.6, 1.2, 4.6)), 2)
    bark_tame = round(float(np.clip(0.14 + bite_excess * 0.24, 0.14, 0.50)), 2)

    configure_persistent_q_precision(
        rig.pq,
        hpf_hz=float(eq_init["hpf_hz"]),  # type: ignore[arg-type]
        band_gains_db=base_gains,
        surgical_notches=[(float(x[0]), float(x[1]), float(x[2])) for x in notches],
        dynamic_mud_thresh_db=dyn_mud_th,
        dynamic_mud_range_db=2.8,
        dynamic_bite_hz=3250.0,
        dynamic_bite_thresh_db=dyn_bite_th,
        dynamic_bite_range_db=dyn_bite_range,
        spectral_hf_hz=7800.0,
        spectral_hf_thresh_db=spec_hf_th,
        spectral_hf_range_db=4.2,
        enable_dynamic_bands=True,
    )
    eq_preview = np.mean(rig.pq(np.stack([v_eval, v_eval], axis=0).astype(np.float32), sr, reset=True), axis=0)

    # --- Stage B: Fast Clean VCA Peak-Tamer + Unity-Gain Optical LA-2A Body Glue Bisection ---
    vca_thresh, vca_gr, c2_param, c2_gr, vca_atk, vca_rel, la2a_gain = rig.calibrate_dynamics_bisection(
        eq_preview, sr=sr, target_crest_db=float(target_profile.vocal_crest_db)
    )

    # --- Stage C: Midrange-Only Saturation & Gentle Pre-Applied Bark Unmasking ---
    dens = init_prof.inst_mid_density_index
    if archetype == "intimate_ballad_sparse":
        sat_amount = round(float(0.9 + dens * 1.0), 1)
        sat_wet_pct = round(float(6.0 + dens * 6.0), 1)
        unmask_dip = round(float(0.65 + dens * 0.55), 2)
    elif archetype == "groovy_mid_pop":
        sat_amount = round(float(1.2 + dens * 1.2), 1)
        sat_wet_pct = round(float(8.0 + dens * 7.0), 1)
        unmask_dip = round(float(0.85 + dens * 0.65), 2)
    else:
        sat_amount = round(float(1.5 + dens * 1.4), 1)
        sat_wet_pct = round(float(10.0 + dens * 8.0), 1)
        unmask_dip = round(float(1.05 + dens * 0.65), 2)

    # Apply Bark Mid Unmasking to anchor evaluation backing BEFORE CMA-ES so level is NEVER double-stacked!
    i_eval_unmasked = apply_dynamic_instrumental_unmasking(i_eval_st, eq_preview, sr=sr, max_dip_db=unmask_dip)
    anchor_inst_unmasked = apply_dynamic_instrumental_unmasking(
        anchor_inst_stereo, anchor_vocal_mono, sr=sr, max_dip_db=unmask_dip
    )

    # Synchronize Early Room level generously enough to seat the vocal INSIDE the backing room
    inst_width_norm = float(np.clip((init_prof.inst_side_mid_db + 16.0) / 10.0, 0.0, 1.0))
    base_er_db = (-16.8 if archetype == "intimate_ballad_sparse" else -18.2) + inst_width_norm * 2.8

    # --- Stage D: 6D Real-VST3 Perceptual Refinement against ALREADY-UNMASKED Backing ---
    # z in [0, 1]^6: [body_tilt, pres_trim, hf_trim, space_trim, soothe_depth_norm, fader_db]
    bounds = np.array([[0.0, 1.0]] * 6, dtype=np.float64)
    prior = np.array([0.50, 0.52, 0.46, 0.55, 0.45, 0.50], dtype=np.float64)

    def _decode(z: np.ndarray) -> VocalChainParams:
        zc = np.clip(z, 0.0, 1.0)
        body_d = (zc[0] - 0.5) * 2.6
        pres_d = (zc[1] - 0.5) * 3.4
        hf_d = (zc[2] - 0.5) * 3.2
        space_off = -3.0 + zc[3] * 6.5  # [-3.0 .. +3.5 dB]
        s2_depth = round(float(1.8 + zc[4] * 2.6), 2)  # [1.8 .. 4.4 dB]

        tuned_gains = dict(base_gains)
        tuned_gains["low_chest"] = round(float(np.clip(base_gains["low_chest"] + body_d * 0.85, -3.2, 3.0)), 2)
        tuned_gains["warm_body"] = round(float(np.clip(base_gains["warm_body"] + body_d, -3.2, 3.2)), 2)
        tuned_gains["vocal_bite"] = round(float(np.clip(base_gains["vocal_bite"] + pres_d, -3.0, 3.2)), 2)
        tuned_gains["clarity_pres"] = round(float(np.clip(base_gains["clarity_pres"] + pres_d * 0.9, -3.0, 3.0)), 2)
        tuned_gains["sibilant_sheen"] = round(float(np.clip(base_gains["sibilant_sheen"] + hf_d * 0.75, -4.0, 1.0)), 2)
        tuned_gains["air_shelf"] = round(float(np.clip(base_gains["air_shelf"] + hf_d, -3.8, 1.8)), 2)

        return VocalChainParams(
            archetype_chain=archetype,
            selected_plugins=list(rig.plugin_names),
            hpf_hz=float(eq_init["hpf_hz"]),  # type: ignore[arg-type]
            body_gain_db=round(0.5 * (tuned_gains["low_chest"] + tuned_gains["warm_body"]), 2),
            mud_cut_db=round(0.5 * (tuned_gains["low_box"] + tuned_gains["mud_honk"]), 2),
            core_mid_gain_db=round(0.5 * (tuned_gains["throat_core"] + tuned_gains["upper_core"]), 2),
            presence_gain_db=round(0.5 * (tuned_gains["vocal_bite"] + tuned_gains["clarity_pres"]), 2),
            air_shelf_db=tuned_gains["air_shelf"],
            vca_threshold_db=vca_thresh,
            vca_measured_gr_db=vca_gr,
            comp2_param=c2_param,
            comp2_measured_gr_db=c2_gr,
            parallel_sat_amount=sat_amount,
            parallel_sat_wet_pct=sat_wet_pct,
            vocal_fader_db=round(float(-8.0 + zc[5] * 16.0), 2),
            early_room_send_db=round(float(base_er_db + space_off * 0.65), 2),
            slap_delay_send_db=round(float(-18.5 + space_off * 0.85), 2),
            tempo_delay_send_db=round(float(-20.2 + space_off * 0.85), 2),
            plate_reverb_send_db=round(float(-19.2 + space_off), 2),
            doubler_send_db=-22.0,
            unmask_mid_dip_db=unmask_dip,
            precision_eq_gains_db=tuned_gains,
            surgical_notches=notches,
            dynamic_mud_thresh_db=dyn_mud_th,
            dynamic_mud_range_db=2.8,
            dynamic_bite_hz=3250.0,
            dynamic_bite_thresh_db=dyn_bite_th,
            dynamic_bite_range_db=dyn_bite_range,
            spectral_hf_hz=7800.0,
            spectral_hf_thresh_db=spec_hf_th,
            spectral_hf_range_db=4.2,
            vca_attack_norm=vca_atk,
            vca_release_norm=vca_rel,
            vca_output_drive=0.1,
            soothe2_depth=s2_depth,
            soothe2_bite_sens=s2_bite_sens,
            soothe2_sib_sens=5.8,
            soothe2_hf_sens=5.0,
            bark_transient_tame_amt=bark_tame,
            lead_core_formant_hz=lead_core_hz,
            la2a_gain_knob=la2a_gain,
        )

    intrinsic_w = {
        "body_warmth": 2.4,
        "boxiness_mud": 2.2,
        "core_mid": 2.4,
        "core_presence": 3.0,
        "sibilance": 2.8,
        "air_sheen": 2.4,
    }
    rolloff_delta = float(np.clip(init_prof.inst_high_rolloff_db - target_profile.inst_high_rolloff_db, -6.0, 6.0))
    tgt_sib_smooth = min(target_profile.vocal_intrinsic_bands_db["sibilance"], -18.8)
    tgt_air_smooth = min(target_profile.vocal_intrinsic_bands_db["air_sheen"] + 0.30 * rolloff_delta, -19.2)

    def _loss(z: np.ndarray) -> float:
        p = _decode(z)
        dry_v = rig.apply_insert_chain(v_eval, sr=sr, p=p, soothe_resolution="eco")
        wet_st = rig.render_spatial_blend(dry_v, sr=sr, p=p)
        bus_st = np.stack([dry_v, dry_v], axis=0) + wet_st
        prof = compute_section_relative_profile(bus_st, i_eval_unmasked, sr=sr)

        # 1. Perceptual Level Blend against ALREADY-UNMASKED backing (eliminates floating overshoot!)
        err_k = ((prof.k_lufs_vi_ratio_db - target_profile.k_lufs_vi_ratio_db) ** 2) * 5.0
        err_mk = ((prof.mid_k_lufs_vi_ratio_db - target_profile.mid_k_lufs_vi_ratio_db) ** 2) * 4.5
        err_pl = ((prof.partial_loudness_margin_db - target_profile.partial_loudness_margin_db) ** 2) * 3.0

        # 2. Intrinsic Vocal Tone Shape (with asymmetric sizzle penalty!)
        err_tone = 0.0
        for b, wt in intrinsic_w.items():
            if b == "sibilance":
                tgt_val = tgt_sib_smooth
            elif b == "air_sheen":
                tgt_val = tgt_air_smooth
            else:
                tgt_val = target_profile.vocal_intrinsic_bands_db[b]
            diff = prof.vocal_intrinsic_bands_db[b] - tgt_val
            if b in ("sibilance", "air_sheen") and diff > 0.0:
                err_tone += wt * (diff**2) * 2.2  # Strong anti-sizzle barrier!
            else:
                err_tone += wt * (diff**2)

        # 3. Song-Specific Backing-Margin Anti-Float Alignment (presence / sibilance / air vs backing!)
        err_margin = 0.0
        for mb, mwt in (("core_presence", 1.1), ("sibilance", 1.5), ("air_sheen", 1.5)):
            m_over = max(0.0, prof.band_margins_db[mb] - (target_profile.band_margins_db[mb] + 0.8))
            err_margin += mwt * (m_over**2)

        # 4. Bidirectional Wet-Tail Cohesion (prevents BOTH muddy reverb AND dry detached floating!)
        tgt_tail = float(np.clip(target_profile.wet_tail_ratio_db, -25.0, -20.5))
        tail_over = max(0.0, prof.wet_tail_ratio_db - (tgt_tail + 1.8))
        tail_dry = max(0.0, (tgt_tail - 2.2) - prof.wet_tail_ratio_db)
        err_tail = (tail_over**2) * 2.5 + (tail_dry**2) * 1.4

        # 5. Syllabic clarity guardrail
        clarity_drop = max(0.0, target_profile.syllabic_clarity_index - prof.syllabic_clarity_index - 0.02)
        err_clarity = (clarity_drop * 35.0) ** 2

        reg = float(np.sum((z - prior) ** 2)) * 0.75
        return float(err_k + err_mk + err_pl + err_tone + err_margin + err_tail + err_clarity + reg)

    optimizer = CMA(mean=prior.copy(), sigma=0.22, bounds=bounds, seed=seed, population_size=6)
    best_z = prior.copy()
    best_loss = _loss(best_z)

    generations = max(5, n_iterations // optimizer.population_size)
    for _ in range(generations):
        sols = []
        for _ in range(optimizer.population_size):
            cand = optimizer.ask()
            val = _loss(cand)
            sols.append((cand, val))
            if val < best_loss:
                best_loss = val
                best_z = cand.copy()
        optimizer.tell(sols)

    best_params = _decode(best_z)
    dry_full = rig.apply_insert_chain(anchor_vocal_mono, sr=sr, p=best_params)
    wet_full = rig.render_spatial_blend(dry_full, sr=sr, p=best_params)
    achieved = compute_section_relative_profile(
        np.stack([dry_full, dry_full], axis=0) + wet_full, anchor_inst_unmasked, sr=sr
    )
    return rig, best_params, achieved, round(best_loss, 3)
