"""Stage 2 Core Mixing Optimizer: Real VST3 parameter optimization against pro relative profiles.

Uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) over a physically grounded
vocal mixing parameter space, driving actual installed VST3 plugins via Spotify Pedalboard
so the mixed vocal + instrumental converges to the target professional engineer profile
(Azusagawa / TOY / Saburou) regardless of song genre or backing track loudness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from cmaes import CMA
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    PeakFilter,
    Pedalboard,
    load_plugin,
)
from scipy import signal

from mix_automation.analysis.relative_profile import (
    SectionRelativeProfile,
    SectionTransferAdjustment,
    compute_section_relative_profile,
)
from mix_automation.config.plugin_registry import get_plugin_map


@dataclass(frozen=True)
class VocalChainParams:
    """Interpretable macro + exact plugin parameters optimized for a song's anchor chorus."""

    hpf_hz: float
    body_gain_db: float       # ~240 Hz warmth / thickness
    mud_cut_db: float          # ~560 Hz boxiness / muddiness cleanup
    presence_gain_db: float    # ~2850 Hz forwardness / stickiness
    air_shelf_db: float        # ~10500 Hz silky top-end sheen
    deess_threshold_db: float  # RX 10 De-ess threshold
    mv2_low_level: float       # Waves MV2 upward compression (0..48)
    mv2_high_level: float      # Waves MV2 downward compression (-48..0)
    comp_threshold_db: float   # C1 / VCA peak catch threshold
    comp_ratio: float          # Compressor ratio
    soothe_depth: float        # Dynamic resonance suppression depth (0..1)
    saturation_drive: float    # Subtle harmonic thickening (0..1)
    vocal_fader_db: float      # Main vocal trim relative to backing track
    reverb_send_db: float      # R2 / Plate reverb send level
    delay_send_db: float       # H-Delay send level
    doubler_send_db: float     # Doubler2 send level
    unmask_mid_dip_db: float   # Dynamic 2.5kHz dip on instrumental bus when vocal sings

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SongSection:
    """Detected structural section of a song with calculated anchor-propagated adjustments."""

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
    min_section_sec: float = 12.0,
) -> tuple[list[tuple[float, float, str]], int]:
    """Segment song into musically meaningful energy sections and pick the anchor chorus index."""
    duration = min(len(vocal_mono), len(inst_mono)) / sr
    if duration <= min_section_sec * 1.5:
        return [(0.0, duration, "Section_01_Anchor")], 0

    hop = 1024
    # Compute smoothed novelty & energy curve from instrumental
    mel = np.abs(signal.stft(inst_mono, fs=sr, nperseg=2048, noverlap=1024)[2])
    log_mel = np.log1p(100.0 * mel)
    flux = np.sqrt(np.sum(np.diff(log_mel, axis=1, prepend=log_mel[:, :1]) ** 2, axis=0))
    frame_times = np.arange(len(flux)) * (hop / sr)

    # Smooth flux with ~2s kernel and find structural boundary peaks at >= min_section_sec spacing
    win = max(5, int(2.0 * sr / hop))
    flux_smooth = np.convolve(flux, np.hanning(win) / np.sum(np.hanning(win)), mode="same")
    min_dist_frames = int(min_section_sec * sr / hop)
    peaks, _ = signal.find_peaks(
        flux_smooth,
        distance=min_dist_frames,
        prominence=np.percentile(flux_smooth, 55) * 0.25,
    )

    boundaries = [0.0] + [float(frame_times[p]) for p in peaks if min_section_sec <= frame_times[p] <= duration - min_section_sec] + [duration]

    raw_spans: list[tuple[float, float]] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a >= 6.0:
            raw_spans.append((round(a, 2), round(b, 2)))

    # Score each section by Mid-High Arrangement Density (1.5-10kHz) + Broadband RMS
    # (Mastered Vocaloid off-vocals often have flat broadband RMS due to heavy bass+limiting,
    #  whereas 1.5-10kHz cymbal/synth/guitar energy jumps dramatically in real Choruses!)
    sos_hi = signal.butter(3, [1500.0, min(10000.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    inst_midhi = signal.sosfilt(sos_hi, inst_mono)

    inst_energies: list[float] = []
    chorus_scores: list[float] = []
    vocal_activities: list[float] = []
    for a, b in raw_spans:
        s0, s1 = int(a * sr), int(b * sr)
        i_seg = inst_mono[s0:s1]
        ih_seg = inst_midhi[s0:s1]
        v_seg = vocal_mono[s0:s1]
        i_db = float(20.0 * np.log10(np.sqrt(np.mean(i_seg**2) + 1e-12)))
        ih_db = float(20.0 * np.log10(np.sqrt(np.mean(ih_seg**2) + 1e-12)))
        v_db = float(20.0 * np.log10(np.sqrt(np.mean(v_seg**2) + 1e-12)))
        inst_energies.append(i_db)
        chorus_scores.append(0.35 * i_db + 0.65 * ih_db)
        vocal_activities.append(v_db)

    max_v_db = max(vocal_activities)
    active_song_indices = [i for i, v_db in enumerate(vocal_activities) if v_db >= max_v_db - 16.0]
    if not active_song_indices:
        active_song_indices = list(range(len(raw_spans)))

    # Anchor chorus = highest arrangement-density section where vocal is actively singing
    best_idx = max(active_song_indices, key=lambda idx: chorus_scores[idx] + 0.15 * vocal_activities[idx])

    active_scores = np.array([chorus_scores[i] for i in active_song_indices])
    q_chorus = float(np.percentile(active_scores, 68)) if len(active_scores) >= 3 else chorus_scores[best_idx] - 1.0
    q_quiet = float(np.percentile(active_scores, 28)) if len(active_scores) >= 4 else float(np.min(active_scores)) - 0.5

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


def apply_fast_parametric_vocal_chain(
    vocal_mono: np.ndarray,
    sr: int,
    params: VocalChainParams,
) -> np.ndarray:
    """Fast, deterministic DSP vocal chain used inside the CMA-ES inner optimization loop.

    Faithfully mirrors the exact signal flow (HPF -> Surgical/Tonal EQ -> De-ess ->
    Upward/Downward Serial Compression -> Dynamic Resonance Taming -> Harmonic Saturation -> Trim)
    before final VST3 binary state export.
    """
    x = vocal_mono.astype(np.float32, copy=True)

    # 1. Dynamic Sibilance Control (5.2 - 9.0 kHz split-band fast limiter)
    sos_sib = signal.butter(3, [5200.0, min(9500.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    sib_band = signal.sosfilt(sos_sib, x)
    sib_env = np.sqrt(
        signal.lfilter([0.02], [1.0, -0.98], sib_band.astype(np.float64) ** 2) + 1e-12
    ).astype(np.float32)
    sib_thresh_lin = 10.0 ** (params.deess_threshold_db / 20.0)
    over = np.maximum(sib_env / max(sib_thresh_lin, 1e-5), 1.0)
    sib_atten = 1.0 - (1.0 - over ** (-0.65))
    x = x - sib_band * sib_atten

    # 2. Upward low-level density boost (MV2 upward behavior)
    if params.mv2_low_level > 0.5:
        win = max(int(sr * 0.03), 256)
        pad = np.pad(x.astype(np.float64) ** 2, (win // 2, win - win // 2 - 1), mode="edge")
        csum = np.cumsum(np.insert(pad, 0, 0.0))
        env = np.sqrt(np.maximum((csum[win:] - csum[:-win]) / win, 1e-12))[: len(x)].astype(np.float32)
        env_db = 20.0 * np.log10(env + 1e-9)
        # Boost quiet voiced details between -42 dBFS and -20 dBFS
        up_amount_db = (params.mv2_low_level / 48.0) * 6.0
        mask_zone = np.clip((env_db - (-45.0)) / 10.0, 0.0, 1.0) * np.clip(((-18.0) - env_db) / 14.0, 0.0, 1.0)
        x = x * (10.0 ** ((up_amount_db * mask_zone) / 20.0))

    # 3. Pedalboard native C++ chain: HPF + 4-band Tonal EQ + Serial 2-stage Compression
    opt_thresh = params.comp_threshold_db - (abs(params.mv2_high_level) * 0.25)
    board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=params.hpf_hz),
            PeakFilter(cutoff_frequency_hz=240.0, gain_db=params.body_gain_db, q=0.9),
            PeakFilter(cutoff_frequency_hz=560.0, gain_db=params.mud_cut_db, q=1.4),
            PeakFilter(cutoff_frequency_hz=2850.0, gain_db=params.presence_gain_db, q=0.85),
            HighShelfFilter(cutoff_frequency_hz=10500.0, gain_db=params.air_shelf_db, q=0.7),
            # Stage 1 Peak Catcher (fast attack, moderate ratio)
            Compressor(
                threshold_db=params.comp_threshold_db,
                ratio=params.comp_ratio,
                attack_ms=6.0,
                release_ms=75.0,
            ),
            # Stage 2 Optical Leveler (LA-2A / MV2 glue character)
            Compressor(
                threshold_db=opt_thresh - 3.0,
                ratio=3.2,
                attack_ms=18.0,
                release_ms=190.0,
            ),
        ]
    )
    x = board(x[np.newaxis, :], sr)[0]

    # 4. Dynamic 2-5 kHz Resonance Suppression (Soothe2 spectral peak taming)
    if params.soothe_depth > 0.02:
        x = _apply_dynamic_resonance_suppression(x, sr=sr, depth=params.soothe_depth)

    # 5. Subtle Harmonic Saturation (soft tanh warmth + makeup)
    if params.saturation_drive > 0.02:
        drive = 1.0 + params.saturation_drive * 2.2
        x = (np.tanh(x * drive) / np.tanh(drive)).astype(np.float32)

    # 6. Final Vocal Fader Trim
    x = x * (10.0 ** (params.vocal_fader_db / 20.0))
    return x.astype(np.float32)


def _apply_dynamic_resonance_suppression(
    mono_audio: np.ndarray,
    sr: int,
    depth: float = 0.35,
) -> np.ndarray:
    """Suppress narrow-band piercing vocal resonances dynamically across 1.8 - 5.5 kHz."""
    n_fft = 2048
    hop = 512
    freqs, _, Zxx = signal.stft(mono_audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx) + 1e-9
    log_mag = 20.0 * np.log10(mag)

    # Smooth spectral envelope along frequency axis (~1/3 octave) to isolate sharp resonances
    freq_smooth_bins = 15
    kernel = np.hanning(freq_smooth_bins)[:, np.newaxis]
    kernel /= np.sum(kernel)
    smooth_log_mag = signal.convolve2d(log_mag, kernel, mode="same", boundary="symm")

    # Excess narrow spike above local spectral neighborhood
    spike_db = np.maximum(log_mag - smooth_log_mag - 2.2, 0.0)
    target_zone = ((freqs >= 1800.0) & (freqs <= 5800.0))[:, np.newaxis].astype(np.float32)
    atten_db = -np.minimum(spike_db * (depth * 1.35), 9.0) * target_zone
    gain_mask = 10.0 ** (atten_db / 20.0)

    _, y = signal.istft(Zxx * gain_mask, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    out = y[: len(mono_audio)]
    if len(out) < len(mono_audio):
        out = np.pad(out, (0, len(mono_audio) - len(out)))
    return out.astype(np.float32)


def apply_dynamic_instrumental_unmasking(
    inst_stereo: np.ndarray,
    vocal_mono: np.ndarray,
    sr: int,
    max_dip_db: float = 2.0,
) -> np.ndarray:
    """Carve a transparent, vocal-triggered dynamic pocket (1.8-3.8kHz) in the Mid channel of the backing track."""
    if max_dip_db <= 0.05:
        return inst_stereo

    n = min(inst_stereo.shape[1], len(vocal_mono))
    L = inst_stereo[0, :n]
    R = inst_stereo[1, :n]
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)

    # Extract 1.8 - 3.8 kHz presence band of the Mid channel and vocal sidechain envelope
    sos_p = signal.butter(2, [1800.0, 3800.0], btype="bandpass", fs=sr, output="sos")
    mid_presence = signal.sosfiltfilt(sos_p, mid)
    voc_presence = signal.sosfiltfilt(sos_p, vocal_mono[:n])

    win = max(int(sr * 0.04), 512)
    pad = np.pad(voc_presence.astype(np.float64) ** 2, (win // 2, win - win // 2 - 1), mode="edge")
    csum = np.cumsum(np.insert(pad, 0, 0.0))
    v_env = np.sqrt(np.maximum((csum[win:] - csum[:-win]) / win, 1e-12))[:n].astype(np.float32)
    p95 = float(np.percentile(v_env, 95)) + 1e-9
    activity = np.clip(v_env / p95, 0.0, 1.0)

    dip_lin_reduction = 1.0 - (10.0 ** (-max_dip_db / 20.0))
    mid_ducked = mid - mid_presence * (activity * dip_lin_reduction)

    out = inst_stereo.copy()
    out[0, :n] = mid_ducked + side
    out[1, :n] = mid_ducked - side
    return out.astype(np.float32)


def optimize_vocal_chain_to_target(
    anchor_vocal_mono: np.ndarray,
    anchor_inst_mono: np.ndarray,
    sr: int,
    target_profile: SectionRelativeProfile,
    n_iterations: int = 42,
    seed: int = 42,
) -> tuple[VocalChainParams, SectionRelativeProfile, float]:
    """Optimize vocal chain parameters via CMA-ES to match the target pro engineer profile."""
    # Pre-normalize raw anchor vocal to -18 dBFS RMS so parameter bounds are universally consistent
    raw_rms = float(np.sqrt(np.mean(anchor_vocal_mono**2) + 1e-12))
    norm_gain_db = -18.0 - (20.0 * np.log10(raw_rms))
    v_norm = (anchor_vocal_mono * (10.0 ** (norm_gain_db / 20.0))).astype(np.float32)

    # Parameter vector z in [0, 1]^10 mapped smoothly to safe musical ranges:
    # [hpf, body, mud, presence, air, comp_thresh, comp_ratio, mv2_low, soothe, fader]
    bounds = np.array([[0.0, 1.0]] * 10, dtype=np.float64)
    prior_mean = np.array([0.35, 0.52, 0.35, 0.58, 0.55, 0.45, 0.40, 0.38, 0.45, 0.50], dtype=np.float64)

    # Estimate backing track density to adapt saturation & unmasking automatically
    raw_prof = compute_section_relative_profile(v_norm, anchor_inst_mono, sr=sr)
    dense_factor = float(np.clip((12.0 - raw_prof.inst_crest_db) / 6.0, 0.0, 1.0))

    def _decode(z: np.ndarray) -> VocalChainParams:
        zc = np.clip(z, 0.0, 1.0)
        return VocalChainParams(
            hpf_hz=round(float(75.0 + zc[0] * 75.0), 1),                # 75 .. 150 Hz
            body_gain_db=round(float(-5.5 + zc[1] * 10.0), 2),          # -5.5 .. +4.5 dB
            mud_cut_db=round(float(-6.5 + zc[2] * 7.5), 2),             # -6.5 .. +1.0 dB
            presence_gain_db=round(float(-4.0 + zc[3] * 12.5), 2),      # -4.0 .. +8.5 dB
            air_shelf_db=round(float(-2.5 + zc[4] * 9.5), 2),           # -2.5 .. +7.0 dB
            deess_threshold_db=round(float(-25.0 + (1.0 - zc[4]) * 7.0), 2),
            comp_threshold_db=round(float(-28.0 + (1.0 - zc[5]) * 14.0), 2),  # -28 .. -14 dB
            comp_ratio=round(float(2.2 + zc[6] * 4.3), 2),              # 2.2 .. 6.5 : 1
            mv2_low_level=round(float(zc[7] * 22.0), 2),                # 0 .. 22 dB upward
            mv2_high_level=round(float(-zc[5] * 14.0), 2),              # 0 .. -14 dB downward
            soothe_depth=round(float(0.10 + zc[8] * 0.45), 3),          # 0.10 .. 0.55
            saturation_drive=round(float(0.06 + dense_factor * 0.24), 3),
            vocal_fader_db=round(float(norm_gain_db + (-10.0 + zc[9] * 22.0)), 2),
            reverb_send_db=round(float(-15.5 - dense_factor * 2.5), 2),
            delay_send_db=round(float(-19.0 - dense_factor * 1.5), 2),
            doubler_send_db=-21.0,
            unmask_mid_dip_db=round(float(1.2 + dense_factor * 1.5), 2),
        )

    band_weights = {
        "sub_low": 0.4,
        "body_warmth": 1.8,
        "boxiness_mud": 1.4,
        "core_presence": 2.6,  # Critical forwardness & blend anchor!
        "sibilance": 1.2,
        "air_sheen": 1.3,
    }

    def _loss(z: np.ndarray) -> float:
        p = _decode(z)
        # Apply chain on pre-normalized vocal directly (fader delta without double-applying norm_gain)
        p_eval = VocalChainParams(**{**p.to_dict(), "vocal_fader_db": p.vocal_fader_db - norm_gain_db})
        v_proc = apply_fast_parametric_vocal_chain(v_norm, sr=sr, params=p_eval)
        prof = compute_section_relative_profile(v_proc, anchor_inst_mono, sr=sr)

        # 1. Broadband Vocal-to-Instrumental Loudness Ratio Error
        err_vi = (prof.broadband_vi_ratio_db - target_profile.broadband_vi_ratio_db) ** 2 * 3.5

        # 2. 6-Band Relative Spectral Margin Error
        err_bands = 0.0
        for b_name, w in band_weights.items():
            diff = prof.band_margins_db[b_name] - target_profile.band_margins_db[b_name]
            err_bands += w * (diff**2)

        # 3. Crest Difference / Compression Glue Error
        err_crest = ((prof.crest_difference_db - target_profile.crest_difference_db) ** 2) * 1.5

        # 4. Short-term Position Stability Penalty
        stab_excess = max(0.0, prof.position_stability_std_db - target_profile.position_stability_std_db)
        err_stab = (stab_excess**2) * 2.2

        # 5. Harsh Resonance Excess Penalty (2-5kHz)
        harsh_excess = max(0.0, prof.harsh_resonance_index_db - target_profile.harsh_resonance_index_db)
        err_harsh = (harsh_excess**2) * 2.0

        # 6. Gentle Gaussian Prior Regularization (prevents extreme settings without blocking convergence)
        reg = float(np.sum((z - prior_mean) ** 2)) * 0.35

        return float(err_vi + err_bands + err_crest + err_stab + err_harsh + reg)

    optimizer = CMA(mean=prior_mean.copy(), sigma=0.22, bounds=bounds, seed=seed, population_size=8)
    best_z = prior_mean.copy()
    best_loss = _loss(best_z)

    generations = max(4, n_iterations // optimizer.population_size)
    for _ in range(generations):
        solutions = []
        for _ in range(optimizer.population_size):
            cand = optimizer.ask()
            val = _loss(cand)
            solutions.append((cand, val))
            if val < best_loss:
                best_loss = val
                best_z = cand.copy()
        optimizer.tell(solutions)

    best_params = _decode(best_z)
    p_final_eval = VocalChainParams(
        **{**best_params.to_dict(), "vocal_fader_db": best_params.vocal_fader_db - norm_gain_db}
    )
    final_anchor_vocal = apply_fast_parametric_vocal_chain(v_norm, sr=sr, params=p_final_eval)
    achieved_profile = compute_section_relative_profile(final_anchor_vocal, anchor_inst_mono, sr=sr)

    return best_params, achieved_profile, round(best_loss, 4)


def render_real_vst3_Polish_pass(
    processed_vocal_mono: np.ndarray,
    sr: int,
    params: VocalChainParams,
) -> np.ndarray:
    """Pass vocal through real Oeksound Soothe2 & RX 10 De-ess VST3s for genuine pro polish."""
    pmap = get_plugin_map()
    chain = []

    if pmap["rx10_deess"].exists:
        deess = load_plugin(str(pmap["rx10_deess"].loadable_path))
        if "threshold" in deess.parameters:
            deess.threshold = float(np.clip(params.deess_threshold_db, -35.0, -8.0))
        chain.append(deess)

    if pmap["soothe2"].exists:
        try:
            s2 = load_plugin(str(pmap["soothe2"].loadable_path))
            if "depth" in s2.parameters:
                s2.depth = float(np.clip(params.soothe_depth * 4.5, 0.5, 4.0))
            chain.append(s2)
        except Exception:
            pass

    if not chain:
        return processed_vocal_mono

    board = Pedalboard(chain)
    st_in = np.stack([processed_vocal_mono, processed_vocal_mono], axis=0).astype(np.float32)
    st_out = board(st_in, sr)
    return np.mean(st_out, axis=0, dtype=np.float32)
