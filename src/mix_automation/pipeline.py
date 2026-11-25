"""v5 End-to-End 44-Song Acoustic k-NN, Continuous Distance Tracker & Native Studio One 5 Pipeline.

Upgrades over v4 addressing every listening & DAW audit finding structurally:
  1. Non-Destructive Pitch & Timing by Default: Never shifts already-tuned/human-grooved takes
     against Vocaloid originals (preventing crammed phrases & lead-vs-harmony desync!). Emits
     high-confidence Studio One 5 markers; `--auto-align` & `--auto-pitch` remain opt-in.
  2. Continuous Backing-Distance & Tone Tracker (`compute_continuous_distance_envelopes`):
     Tracks 350Hz-7.5kHz competition energy continuously (catching sub-bass-heavy quiet breakdowns
     like 2:15 even when broadband RMS stays high!) and smoothly automates vocal gain, parallel
     saturation, upper-mid edge softening, and room cohesion so distance never detaches.
  3. True Stereo K-LUFS Harmony Bus + Real-Harmony Priority: Fixes v4's Mid-only M/S gain overshoot,
     tames harmony sibilants (`4.8kHz+`), glues harmonies in shared Early Room + Plate, and keeps
     `Nectar 3` muted as recording guides whenever the user already recorded real harmonies!
  4. Native Steinberg `.vstpreset` Unwrapping (`extract_steinberg_comp_and_cont`): Decodes
     Pedalboard's JUCE `<VST3PluginState>` wrapper so EVERY plugin in Studio One 5 loads with its
     EXACT optimized knobs, clean harmony racks, expanded `メイン` inserts, and matching levels!
  5. Flat-Folder Auto-Discovery (`Fall in Love` style) + Personalized Azusagawa Taste Anchor
     (`--taste azusagawa --taste-weight 0.25` default on top of 75% 44-song acoustic k-NN).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import librosa
import numpy as np
from pedalboard import Compressor, HighpassFilter, LowpassFilter, PeakFilter, Pedalboard
from scipy import signal

from mix_automation.analysis.reference_extractor import (
    BSRoFormerSeparator,
    synthesize_acoustic_knn_target,
)
from mix_automation.analysis.relative_profile import (
    SectionRelativeProfile,
    classify_backing_archetype,
    compute_anchor_propagation,
    compute_backing_full_descriptors,
    compute_section_relative_profile,
)
from mix_automation.creative.fx_and_harmony import (
    apply_verified_radio_fx_inline,
    detect_creative_fx_events,
    generate_nectar3_harmonies_and_plan,
    render_ethereal_throw_stem,
)
from mix_automation.dsp.pitch_timing import ReviewMarker
from mix_automation.dsp.preprocess import (
    PreprocessConfig,
    preprocess_vocal_pipeline,
    stage_vocal_loudness,
)
from mix_automation.dsp.vst3_optimizer import (
    SongSection,
    apply_2mix_bus_glue_and_safe_master,
    apply_dynamic_distance_tone_soften,
    apply_dynamic_instrumental_unmasking,
    compute_continuous_distance_envelopes,
    detect_song_sections,
    optimize_real_vst3_chain,
    process_dedicated_harmony_bus,
)
from mix_automation.io.audio_loader import (
    _classify_filename_role,
    consolidate_vocal_stem_folder,
    load_audio,
)
from mix_automation.io.song_builder import (
    TrackMediaSpec,
    build_studio_one_song,
    export_template_vst3_states,
    write_pcm24_wav,
)


def _dict_to_profile(d: dict, resolved_name: str) -> SectionRelativeProfile:
    default_intrinsic = {
        "sub_low": -35.0,
        "body_warmth": -6.3,
        "boxiness_mud": -4.1,
        "core_mid": -6.3,
        "core_presence": -8.8,
        "sibilance": -17.5,
        "air_sheen": -19.0,
    }
    return SectionRelativeProfile(
        vocal_rms_db=d["vocal_rms_db"],
        inst_rms_db=d["inst_rms_db"],
        broadband_vi_ratio_db=d["broadband_vi_ratio_db"],
        k_lufs_vi_ratio_db=d.get("k_lufs_vi_ratio_db", 0.5),
        mid_k_lufs_vi_ratio_db=d.get("mid_k_lufs_vi_ratio_db", 1.2),
        partial_loudness_margin_db=d.get("partial_loudness_margin_db", 1.6),
        vocal_crest_db=d["vocal_crest_db"],
        inst_crest_db=d["inst_crest_db"],
        crest_difference_db=d["crest_difference_db"],
        position_stability_std_db=d["position_stability_std_db"],
        harsh_resonance_index_db=d["harsh_resonance_index_db"],
        syllabic_clarity_index=d.get("syllabic_clarity_index", 0.19),
        wet_tail_ratio_db=d.get("wet_tail_ratio_db", -22.0),
        vocal_side_mid_db=d.get("vocal_side_mid_db", -10.5),
        inst_side_mid_db=d.get("inst_side_mid_db", -7.5),
        inst_mid_density_index=d.get("inst_mid_density_index", 0.58),
        inst_high_rolloff_db=d.get("inst_high_rolloff_db", -8.8),
        backing_archetype=d.get("backing_archetype", resolved_name),
        vocal_intrinsic_bands_db=d.get("vocal_intrinsic_bands_db", default_intrinsic),
        band_margins_db=d["band_margins_db"],
        inst_band_levels_db=d["inst_band_levels_db"],
        active_frame_ratio=d["active_frame_ratio"],
    )


def load_target_profile(
    profiles_json: Path,
    style: str = "auto",
    vocal_median_f0_hz: float = 260.0,
    backing_archetype: str = "dense_rock_edm",
    query_chorus_prof: SectionRelativeProfile | None = None,
    query_full_desc: dict[str, float] | None = None,
    taste_style: str | None = None,
    taste_weight: float = 0.0,
) -> tuple[SectionRelativeProfile, str, list[dict], float]:
    """Load the v5 continuous acoustic k-NN or explicit engineer target profile (`schema_version` 4.0)."""
    data = json.loads(profiles_json.read_text(encoding="utf-8"))
    targets = data["target_profiles"]
    songs = data.get("songs", [])
    req = style.lower().strip()

    taste_prof: SectionRelativeProfile | None = None
    ts_clean = (taste_style or "").lower().strip()
    if ts_clean and ts_clean not in {"none", "off", "false"} and taste_weight > 0.01:
        if ts_clean in targets:
            taste_prof = _dict_to_profile(targets[ts_clean], ts_clean)

    if req == "auto" and songs and query_chorus_prof is not None and query_full_desc is not None:
        knn_prof, neighbors, verse_shift = synthesize_acoustic_knn_target(
            songs_records=songs,
            query_chorus_prof=query_chorus_prof,
            query_full_desc=query_full_desc,
            query_median_f0_hz=vocal_median_f0_hz,
            k=9,
            taste_profile=taste_prof,
            taste_weight=taste_weight if taste_prof is not None else 0.0,
        )
        label = f"knn_44_{backing_archetype}"
        if taste_prof is not None and taste_weight > 0.01:
            label += f"+{ts_clean}({int(taste_weight * 100)}%)"
        return knn_prof, label, neighbors, verse_shift

    reg_key = "low_mid" if vocal_median_f0_hz < 295.0 else "high"
    stratified_key = f"{reg_key}_{backing_archetype}"
    reg_fallback_key = f"register_{reg_key}"

    if req == "auto":
        bucket_n = sum(
            1 for s in songs if s.get("vocal_register") == reg_key and s.get("backing_archetype") == backing_archetype
        )
        if stratified_key in targets and (bucket_n >= 3 or not songs):
            resolved = stratified_key
        elif backing_archetype in targets:
            resolved = backing_archetype
        elif reg_fallback_key in targets:
            resolved = reg_fallback_key
        else:
            resolved = "consensus"
    else:
        if req not in targets:
            raise ValueError(f"Style {style!r} not found. Available: {['auto'] + list(targets.keys())}")
        resolved = req

    prof = _dict_to_profile(targets[resolved], resolved)
    return prof, resolved, [], 0.55


def discover_song_assets(song_folder: Path) -> dict[str, Path | list[Path] | None]:
    """Automatically identify stems (supporting both `stems/` subfolder & flat folders), off-vocal, original & self-mix."""
    stems_dir = song_folder / "stems"
    stem_files = sorted(stems_dir.glob("*.wav")) if stems_dir.exists() else []

    inst_file: Path | None = None
    orig_file: Path | None = None
    self_mix_file: Path | None = None
    lyrics_file: Path | None = None
    flat_stem_candidates: list[Path] = []

    for f in sorted(song_folder.iterdir()):
        if f.is_dir():
            continue
        low = f.name.lower()
        suf = f.suffix.lower()
        if suf == ".txt":
            lyrics_file = f
        elif "自mix" in low and "オケなし" not in low:
            self_mix_file = f
        elif any(k in low for k in ("inst", "off vocal", "offvocal", "off_vocal", "オケ")) and "自mix" not in low:
            inst_file = f
        elif suf == ".wav" and _classify_filename_role(f.stem) is not None:
            flat_stem_candidates.append(f)
        elif suf in {".mp3", ".wav", ".webm", ".flac", ".m4a"} and "自mix" not in low:
            orig_file = f

    if not stem_files and flat_stem_candidates:
        stem_files = flat_stem_candidates

    if inst_file is None:
        raise FileNotFoundError(f"Could not auto-detect off-vocal/INST file in {song_folder}")
    if not stem_files:
        raise FileNotFoundError(f"No vocal stem WAVs found inside {song_folder} or {stems_dir}")

    return {
        "stems": stem_files,
        "inst": inst_file,
        "original": orig_file,
        "self_mix": self_mix_file,
        "lyrics": lyrics_file,
    }


def _align_reference_vocal_to_offvocal(
    ref_vocal_mono: np.ndarray,
    ref_inst_mono: np.ndarray,
    user_inst_mono: np.ndarray,
    sr: int,
    target_len: int,
    max_offset_sec: float = 12.0,
) -> tuple[np.ndarray, float]:
    """Time-lock separated original vocal to the user's off-vocal within 10ms accuracy."""
    hop = int(sr * 0.01)
    scan_len = min(len(ref_inst_mono), len(user_inst_mono), sr * 90)
    n_f = scan_len // hop
    if n_f < 100:
        out = np.zeros(target_len, dtype=np.float32)
        c = min(target_len, len(ref_vocal_mono))
        out[:c] = ref_vocal_mono[:c]
        return out, 0.0

    u_env = np.sqrt(np.mean(user_inst_mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1))
    r_env = np.sqrt(np.mean(ref_inst_mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1))
    u_diff = np.maximum(np.diff(u_env, prepend=0.0), 0.0)
    r_diff = np.maximum(np.diff(r_env, prepend=0.0), 0.0)

    corr = signal.correlate(u_diff, r_diff, mode="full")
    lags = signal.correlation_lags(len(u_diff), len(r_diff), mode="full")
    max_lag_f = int(max_offset_sec / 0.01)
    valid_m = np.abs(lags) <= max_lag_f
    best_lag_f = int(lags[valid_m][int(np.argmax(corr[valid_m]))])
    shift_samples = best_lag_f * hop

    aligned = np.zeros(target_len, dtype=np.float32)
    if shift_samples >= 0:
        copy_n = min(target_len - shift_samples, len(ref_vocal_mono))
        if copy_n > 0:
            aligned[shift_samples : shift_samples + copy_n] = ref_vocal_mono[:copy_n]
    else:
        src_start = -shift_samples
        copy_n = min(target_len, max(0, len(ref_vocal_mono) - src_start))
        if copy_n > 0:
            aligned[:copy_n] = ref_vocal_mono[src_start : src_start + copy_n]

    return aligned, round(best_lag_f * 0.01, 3)


def _smooth_envelope(raw_env: np.ndarray, sr: int, smooth_sec: float = 0.25) -> np.ndarray:
    win_len = max(16, int(smooth_sec * sr))
    xf_win = np.hanning(win_len).astype(np.float32)
    xf_win /= np.sum(xf_win)
    return signal.fftconvolve(raw_env.astype(np.float32), xf_win, mode="same").astype(np.float32)


def run_full_automix_for_song(
    song_folder: Path,
    template_song_path: Path,
    profiles_json_path: Path,
    output_root: Path,
    style: str = "auto",
    target_sr: int = 44100,
    force_auto_pitch: bool = False,
    enable_auto_align: bool = False,
    taste_style: str | None = "azusagawa",
    taste_weight: float = 0.25,
    enable_nectar_in_mix: bool = False,
) -> dict:
    """Execute the complete v5 automated vocal mixing & true 1:1 Studio One 5 pipeline."""
    song_name = song_folder.name
    safe_slug = song_name.replace("&", "and").replace(" ", "_")
    ts_norm = (taste_style or "").lower().strip()
    eff_taste = None if ts_norm in {"", "none", "off", "false"} else ts_norm
    eff_tw = taste_weight if eff_taste else 0.0

    out_dir = output_root / f"{safe_slug}_{style}"
    media_dir = out_dir / "Media"
    media_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [1/7] Loading, acoustically deduplicating & classifying stems for '{song_name}' ===", flush=True)
    assets = discover_song_assets(song_folder)
    inst_stereo, _ = load_audio(assets["inst"], target_sr=target_sr, mono=False)  # type: ignore[arg-type]
    inst_mono = np.mean(inst_stereo, axis=0, dtype=np.float32)

    consolidated = consolidate_vocal_stem_folder(assets["stems"], target_sr=target_sr)  # type: ignore[arg-type]
    print(
        f"Stem Roles: { {k: len(v) for k, v in consolidated.stem_roles_summary.items()} } | "
        f"Deduplicated Duplicates Skipped: {len(consolidated.deduplicated_files)}",
        flush=True,
    )

    common_len = max(inst_stereo.shape[1], len(consolidated.lead_mono))
    if inst_stereo.shape[1] < common_len:
        inst_stereo = np.pad(inst_stereo, ((0, 0), (0, common_len - inst_stereo.shape[1])))
        inst_mono = np.pad(inst_mono, (0, common_len - len(inst_mono)))
    lead_raw = np.pad(consolidated.lead_mono, (0, max(0, common_len - len(consolidated.lead_mono))))[:common_len]

    tempo_est, _ = librosa.beat.beat_track(y=inst_mono[: min(len(inst_mono), target_sr * 90)], sr=target_sr)
    bpm = round(float(np.atleast_1d(tempo_est)[0]), 1)
    if bpm < 75.0:
        bpm *= 2.0

    ref_vocal_aligned: np.ndarray | None = None
    orig_align_offset_sec = 0.0
    sep_cache_root = output_root.parent / "processed"
    if assets["original"] is not None:
        try:
            orig_sep = BSRoFormerSeparator(
                cache_dir=sep_cache_root / "orig_stems",
                model_dir=sep_cache_root / "roformer_models",
            )
            ov_st, oi_st, _, _ = orig_sep.separate(
                assets["original"], slug=f"orig_{safe_slug}", target_sr=target_sr  # type: ignore[arg-type]
            )
            ov_m = np.mean(ov_st, axis=0, dtype=np.float32)
            oi_m = np.mean(oi_st, axis=0, dtype=np.float32)
            ref_vocal_aligned, orig_align_offset_sec = _align_reference_vocal_to_offvocal(
                ov_m, oi_m, inst_mono, sr=target_sr, target_len=common_len
            )
            print(f"Original reference vocal separated & aligned (offset={orig_align_offset_sec:+.2f}s)", flush=True)
        except Exception as exc:
            print(f"[Warning] Could not separate original reference vocal: {exc}", flush=True)
            ref_vocal_aligned = None

    print(f"Detected Tempo: {bpm} BPM | Duration: {common_len / target_sr:.2f}s", flush=True)

    print("=== [2/7] Stage 1 Groove-Safe Pre-processing (RX10 Music + Spectral De-ess >5.8kHz + -18 LUFS) ===", flush=True)
    prep_cfg = PreprocessConfig(force_auto_pitch=force_auto_pitch, enable_auto_align=enable_auto_align)
    lead_prepared, pt_report, scale_info, prep_calib = preprocess_vocal_pipeline(
        lead_raw,
        inst_mono,
        sr=target_sr,
        ref_vocal_mono=ref_vocal_aligned,
        cfg=prep_cfg,
    )
    _, key_root, key_mode = scale_info
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    detected_key_str = f"{note_names[key_root]} {key_mode}"
    print(
        f"Key: {detected_key_str} | F0: {pt_report.median_f0_hz}Hz | Melodyne: {pt_report.already_melodyne_tuned} "
        f"(slipped={pt_report.phrases_slipped_count}, pitchNudged={pt_report.notes_nudged_count}) | "
        f"De-ess: th={prep_calib.deess_calibrated_threshold_db:+.1f}dB (sibGR={prep_calib.measured_sibilant_gr_db:.2f}dB, "
        f"attackDelta={prep_calib.measured_attack_band_delta_db:+.2f}dB) | Staged: {prep_calib.staged_lufs} LUFS",
        flush=True,
    )

    print("=== [3/7] 44-Song Acoustic k-NN (+Taste) & 3-Stage Hybrid Real-VST3 Optimization ===", flush=True)
    raw_spans, anchor_idx = detect_song_sections(lead_prepared, inst_mono, sr=target_sr)
    a_start, a_end, a_lbl = raw_spans[anchor_idx]
    s0, s1 = int(a_start * target_sr), int(a_end * target_sr)

    init_anchor_prof = compute_section_relative_profile(lead_prepared[s0:s1], inst_stereo[:, s0:s1], sr=target_sr)
    full_backing_desc = compute_backing_full_descriptors(inst_stereo, sr=target_sr)
    song_archetype = classify_backing_archetype(
        inst_crest_db=init_anchor_prof.inst_crest_db,
        inst_band_levels_db=init_anchor_prof.inst_band_levels_db,
        full_inst_mid_density_index=full_backing_desc["full_inst_mid_density_index"],
        backing_dynamic_range_db=full_backing_desc["backing_dynamic_range_db"],
        full_inst_crest_db=full_backing_desc["full_inst_crest_db"],
        chorus_inst_mid_density_index=init_anchor_prof.inst_mid_density_index,
    )

    target_prof, resolved_style, knn_neighbors, learned_verse_shift = load_target_profile(
        profiles_json_path,
        style=style,
        vocal_median_f0_hz=pt_report.median_f0_hz,
        backing_archetype=song_archetype,
        query_chorus_prof=init_anchor_prof,
        query_full_desc=full_backing_desc,
        taste_style=eff_taste,
        taste_weight=eff_tw,
    )

    rig, chain_params, achieved_anchor_prof, final_loss = optimize_real_vst3_chain(
        lead_prepared[s0:s1],
        inst_stereo[:, s0:s1],
        sr=target_sr,
        bpm=bpm,
        target_profile=target_prof,
        n_iterations=32,
        forced_archetype=song_archetype,
    )
    top_nn_str = ", ".join(f"{n['title']}({n['weight_pct']}%)" for n in knn_neighbors[:3]) if knn_neighbors else resolved_style
    print(
        f"Archetype: '{chain_params.archetype_chain}' (FullDens={full_backing_desc['full_inst_mid_density_index']}, "
        f"DynR={full_backing_desc['backing_dynamic_range_db']}dB) | Profile: '{resolved_style}' | k-NN: [{top_nn_str}]\n"
        f"  Anchor '{a_lbl}' (loss={final_loss}) | Sections: {len(raw_spans)} | Verse Intimacy Shift: +{learned_verse_shift:.2f}dB\n"
        f"  EQ: HPF={chain_params.hpf_hz}Hz, Body={chain_params.body_gain_db:+.2f}, Mud={chain_params.mud_cut_db:+.2f}, "
        f"Mid={chain_params.core_mid_gain_db:+.2f}, Pres={chain_params.presence_gain_db:+.2f}, Air={chain_params.air_shelf_db:+.2f} dB\n"
        f"  Dyn & Texture: VCA th={chain_params.vca_threshold_db}dB (GR={chain_params.vca_measured_gr_db}dB), "
        f"Comp2={chain_params.comp2_param} (crestGR={chain_params.comp2_measured_gr_db}dB), "
        f"Sat={chain_params.parallel_sat_amount} ({chain_params.parallel_sat_wet_pct}% par), Fader={chain_params.vocal_fader_db:+.2f}dB",
        flush=True,
    )

    print("=== [4/7] Continuous Backing-Distance Tracker, Dynamic Tone Soften & Section Propagation ===", flush=True)
    # Continuous 350Hz-7.5kHz competition tracker: locks vocal distance seamlessly across sudden drops (e.g. 2:15)!
    dist_envs = compute_continuous_distance_envelopes(
        inst_stereo, sr=target_sr, anchor_span=(s0, s1), verse_intimacy_shift_db=learned_verse_shift
    )

    sat_env_sec = np.ones(common_len, dtype=np.float32)
    space_env_sec = np.ones(common_len, dtype=np.float32)
    delay_env_sec = np.ones(common_len, dtype=np.float32)
    for st_sec, en_sec, lbl in raw_spans:
        i0, i1 = int(st_sec * target_sr), int(en_sec * target_sr)
        if "Quiet_Bridge" in lbl:
            sat_env_sec[i0:i1] = 0.72
            delay_env_sec[i0:i1] = 0.80
            space_env_sec[i0:i1] = 1.12
        elif "Verse_PreChorus" in lbl:
            sat_env_sec[i0:i1] = 0.88
            delay_env_sec[i0:i1] = 0.88
            space_env_sec[i0:i1] = 0.96
        elif lbl.endswith("_Chorus"):
            sat_env_sec[i0:i1] = 1.00
            delay_env_sec[i0:i1] = 1.00
            space_env_sec[i0:i1] = 1.05

    # Multiply section envelopes by continuous Mid-High distance envelopes for seamless intro-to-outro cohesion!
    sat_env = np.clip(_smooth_envelope(sat_env_sec, target_sr, 0.35) * dist_envs["sat_env"], 0.35, 1.05)
    space_env = np.clip(_smooth_envelope(space_env_sec, target_sr, 0.35) * dist_envs["space_env"], 0.85, 1.40)
    delay_env = np.clip(_smooth_envelope(delay_env_sec, target_sr, 0.35), 0.70, 1.10)

    lead_chained_base = rig.apply_insert_chain(lead_prepared, sr=target_sr, p=chain_params, sat_envelope=sat_env)

    song_sections: list[SongSection] = []
    section_envelope = np.ones(common_len, dtype=np.float32)

    for idx, (st_sec, en_sec, lbl) in enumerate(raw_spans):
        i0, i1 = int(st_sec * target_sr), int(en_sec * target_sr)
        sec_prof = compute_section_relative_profile(lead_chained_base[i0:i1], inst_stereo[:, i0:i1], sr=target_sr)
        if idx == anchor_idx or "Instrumental" in lbl:
            song_sections.append(
                SongSection(
                    index=idx,
                    label=lbl,
                    start_sec=st_sec,
                    end_sec=en_sec,
                    is_anchor=(idx == anchor_idx),
                    inst_rms_db=sec_prof.inst_rms_db,
                    clip_gain_db=0.0,
                    adjustment=None,
                )
            )
        else:
            adj = compute_anchor_propagation(achieved_anchor_prof, sec_prof)
            intimacy_bonus = 0.0
            if "Quiet_Bridge" in lbl:
                intimacy_bonus = learned_verse_shift * 0.55
            elif "Verse_PreChorus" in lbl:
                intimacy_bonus = learned_verse_shift * 0.40
            cg_db = round(float(np.clip(adj.input_gain_delta_db * 0.55 + intimacy_bonus, -3.2, 2.8)), 2)
            song_sections.append(
                SongSection(
                    index=idx,
                    label=lbl,
                    start_sec=st_sec,
                    end_sec=en_sec,
                    is_anchor=False,
                    inst_rms_db=sec_prof.inst_rms_db,
                    clip_gain_db=cg_db,
                    adjustment=adj,
                )
            )
            section_envelope[i0:i1] = 10.0 ** (cg_db / 20.0)

    section_envelope = _smooth_envelope(section_envelope, target_sr, 0.16)
    # Apply BOTH section clip-gain envelope AND continuous distance gain + upper-mid softening on quiet drops!
    lead_final_dry = (lead_chained_base * section_envelope * dist_envs["gain_env"]).astype(np.float32)
    lead_final_dry = apply_dynamic_distance_tone_soften(
        lead_final_dry, sr=target_sr, soften_env=dist_envs["soften_env"], max_cut_db=1.6
    )

    print("=== [5/7] Stage 3 Reference-Verified FX & Stage 4 Nectar 3 Harmonies (Real-Harmony Priority) ===", flush=True)
    fx_events = detect_creative_fx_events(
        lead_final_dry, inst_mono, sr=target_sr, ref_vocal_mono=ref_vocal_aligned
    )
    lead_final_dry = apply_verified_radio_fx_inline(lead_final_dry, sr=target_sr, events=fx_events)
    fx_throw_stereo = render_ethereal_throw_stem(lead_final_dry, sr=target_sr, events=fx_events, bpm=bpm)

    harmony_res = generate_nectar3_harmonies_and_plan(
        lead_prepared,
        inst_mono,
        sr=target_sr,
        sections=song_sections,
        existing_user_harmonies=(
            [consolidated.harmonies_stereo_bus] if consolidated.harmonies_stereo_bus is not None else None
        ),
    )
    print(
        f"Verified FX Cues: {len(fx_events)} | Harmony Engine: {harmony_res.engine_used} "
        f"({harmony_res.total_harmony_notes} notes, {harmony_res.chord_clash_notes_rescued} chord clashes rescued) | "
        f"User Harmonies Present: {consolidated.has_user_harmonies}",
        flush=True,
    )

    print("=== [6/7] Bark Spectral Unmasking, True Stereo K-LUFS Harmony Bus & Closed-Loop QA ===", flush=True)
    spatial_wet_stereo = rig.render_spatial_blend(
        lead_final_dry, sr=target_sr, p=chain_params, section_space_env=space_env, section_delay_env=delay_env
    )

    lead_active = lead_final_dry[np.abs(lead_final_dry) > 1e-3]
    lead_active_rms = float(np.sqrt(np.mean(lead_active**2) + 1e-12)) if lead_active.size > 0 else 0.05

    def _balance_aux_stereo(aux_st: np.ndarray | None, rel_db: float, hpf: float = 190.0) -> np.ndarray | None:
        if aux_st is None:
            return None
        padded = np.zeros((2, common_len), dtype=np.float32)
        n_c = min(common_len, aux_st.shape[1])
        padded[:, :n_c] = aux_st[:, :n_c]
        nz = padded[np.abs(padded) > 1e-4]
        if nz.size < 1000:
            return None
        b = Pedalboard(
            [
                HighpassFilter(hpf),
                PeakFilter(cutoff_frequency_hz=2600.0, gain_db=-3.2, q=1.0),
                LowpassFilter(8200.0),
                Compressor(threshold_db=-23.0, ratio=4.0, attack_ms=8.0, release_ms=100.0),
            ]
        )
        proc = b(padded, target_sr)
        cur_rms = float(np.sqrt(np.mean(proc[np.abs(proc) > 1e-4] ** 2) + 1e-12))
        target_rms = lead_active_rms * (10.0 ** (rel_db / 20.0))
        gain = float(np.clip(target_rms / cur_rms, 0.04, 16.0))
        return (proc * gain).astype(np.float32)

    aux_layers_stereo = np.zeros((2, common_len), dtype=np.float32)
    harm_bus_processed = np.zeros((2, common_len), dtype=np.float32)
    harm1_mono_for_s1: np.ndarray | None = None
    harm2_mono_for_s1: np.ndarray | None = None
    dbl_bal: np.ndarray | None = None
    sub_oc_bal: np.ndarray | None = None
    sup_oc_bal: np.ndarray | None = None

    if consolidated.double_stereo is not None:
        dbl_bal = _balance_aux_stereo(consolidated.double_stereo, rel_db=-11.8, hpf=200.0)
        if dbl_bal is not None:
            aux_layers_stereo += dbl_bal

    if consolidated.sub_octave_stereo is not None:
        sub_oc_bal = _balance_aux_stereo(consolidated.sub_octave_stereo, rel_db=-13.8, hpf=110.0)
        if sub_oc_bal is not None:
            aux_layers_stereo += sub_oc_bal

    if consolidated.super_octave_stereo is not None:
        sup_oc_bal = _balance_aux_stereo(consolidated.super_octave_stereo, rel_db=-14.5, hpf=260.0)
        if sup_oc_bal is not None:
            aux_layers_stereo += sup_oc_bal

    harmony_target_rel_db = -11.2
    if consolidated.harmonies_stereo_bus is not None:
        harm_bus_processed = process_dedicated_harmony_bus(
            consolidated.harmonies_stereo_bus,
            lead_final_dry,
            sr=target_sr,
            target_rel_db=harmony_target_rel_db,
            side_boost_db=1.4,
            plate_plugin=rig.r2,
            room_plugin=rig.phoenix,
        )
        # Stage Studio One harmvocal / harmvocal2 mono stems cleanly to -18 LUFS matching mainvocal!
        if consolidated.harm1_mono is not None:
            h1_pad = np.pad(consolidated.harm1_mono, (0, max(0, common_len - len(consolidated.harm1_mono))))[:common_len]
            harm1_mono_for_s1, _ = stage_vocal_loudness(h1_pad, sr=target_sr, target_lufs=-18.0)
        if consolidated.harm2_mono is not None:
            h2_pad = np.pad(consolidated.harm2_mono, (0, max(0, common_len - len(consolidated.harm2_mono))))[:common_len]
            harm2_mono_for_s1, _ = stage_vocal_loudness(h2_pad, sr=target_sr, target_lufs=-18.0)

    # Real-Harmony Priority: ONLY blend Nectar 3 into 2mix when user has ZERO harmonies (or explicitly enabled)!
    use_nectar_in_2mix = (not consolidated.has_user_harmonies) or enable_nectar_in_mix
    if use_nectar_in_2mix:
        harm_bus_processed = (harm_bus_processed + harmony_res.auto_mix_harmony_stereo * 0.72).astype(np.float32)

    # 14-Band Bark Adaptive Spectral Mid-Channel Unmasking (Side stereo image 100% intact!)
    inst_unmasked = apply_dynamic_instrumental_unmasking(
        inst_stereo, lead_final_dry, sr=target_sr, max_dip_db=chain_params.unmask_mid_dip_db
    )

    # --- Perceptual Closed-Loop Post-Sum Vocal Bus QA (Wet Tail + Bidirectional Stereo Width + K-LUFS) ---
    wet_total = spatial_wet_stereo + fx_throw_stereo
    lead_stereo = np.stack([lead_final_dry, lead_final_dry], axis=0)
    vocal_bus_stereo = lead_stereo + aux_layers_stereo + harm_bus_processed + wet_total

    pre_qa_anchor = compute_section_relative_profile(
        vocal_bus_stereo[:, s0:s1], inst_unmasked[:, s0:s1], sr=target_sr
    )

    # 1. Trim wet space if excessive
    wet_ceiling_db = max(target_prof.wet_tail_ratio_db + 1.8, -18.5)
    if pre_qa_anchor.wet_tail_ratio_db > wet_ceiling_db:
        wet_cut_db = max(-8.0, wet_ceiling_db - pre_qa_anchor.wet_tail_ratio_db)
        wet_total = wet_total * float(10.0 ** (wet_cut_db / 20.0))
        vocal_bus_stereo = lead_stereo + aux_layers_stereo + harm_bus_processed + wet_total
        pre_qa_anchor = compute_section_relative_profile(
            vocal_bus_stereo[:, s0:s1], inst_unmasked[:, s0:s1], sr=target_sr
        )

    # 2. Bidirectional Stereo Harmony QA: tame overly loud/wide harmonies OR gently bloom overly narrow ones!
    harm_qa_trim_db = 0.0
    if float(np.max(np.abs(harm_bus_processed))) > 1e-4:
        side_excess = pre_qa_anchor.vocal_side_mid_db - target_prof.vocal_side_mid_db
        if side_excess > 1.4:
            # Harmonies/choir are poking out too wide/loud -> pull harmony bus down smoothly into pro pocket!
            harm_qa_trim_db = round(-float(np.clip(side_excess * 0.85, 0.8, 5.5)), 2)
            harm_bus_processed = (harm_bus_processed * float(10.0 ** (harm_qa_trim_db / 20.0))).astype(np.float32)
            vocal_bus_stereo = lead_stereo + aux_layers_stereo + harm_bus_processed + wet_total
            pre_qa_anchor = compute_section_relative_profile(
                vocal_bus_stereo[:, s0:s1], inst_unmasked[:, s0:s1], sr=target_sr
            )
        elif side_excess < -2.6:
            width_gap = min(2.8, (-1.8) - side_excess)
            hm = 0.5 * (harm_bus_processed[0] + harm_bus_processed[1])
            hs = 0.5 * (harm_bus_processed[0] - harm_bus_processed[1]) * float(10.0 ** (width_gap / 20.0))
            harm_bus_processed = np.stack([hm + hs, hm - hs], axis=0).astype(np.float32)
            vocal_bus_stereo = lead_stereo + aux_layers_stereo + harm_bus_processed + wet_total
            pre_qa_anchor = compute_section_relative_profile(
                vocal_bus_stereo[:, s0:s1], inst_unmasked[:, s0:s1], sr=target_sr
            )

    # 3. Final Perceptual Level Lock (K-LUFS + Mid K-LUFS + Partial Loudness)
    perceptual_overshoot_db = (
        0.45 * (pre_qa_anchor.k_lufs_vi_ratio_db - target_prof.k_lufs_vi_ratio_db)
        + 0.35 * (pre_qa_anchor.mid_k_lufs_vi_ratio_db - target_prof.mid_k_lufs_vi_ratio_db)
        + 0.20 * (pre_qa_anchor.partial_loudness_margin_db - target_prof.partial_loudness_margin_db)
    )
    closed_loop_trim_db = 0.0
    if abs(perceptual_overshoot_db) > 0.35:
        closed_loop_trim_db = round(float(np.clip(-perceptual_overshoot_db * 0.85, -3.5, 4.0)), 2)
        vocal_bus_stereo = (vocal_bus_stereo * float(10.0 ** (closed_loop_trim_db / 20.0))).astype(np.float32)

    final_master_2mix = apply_2mix_bus_glue_and_safe_master(
        inst_unmasked, vocal_bus_stereo, sr=target_sr, target_lufs=-9.0
    )

    inst_raw_wav = media_dir / "00_offvocal_raw_backup.wav"
    inst_wav = media_dir / "01_offvocal_unmasked.wav"
    main_prep_wav = media_dir / "02_mainvocal_prepared.wav"
    harm1_wav = media_dir / "03_harmvocal_1.wav"
    harm2_wav = media_dir / "04_harmvocal_2.wav"
    throws_wav = media_dir / "05_fx_verified_throws.wav"
    nectar_up_wav = media_dir / "06_nectar3_harm_upper.wav"
    nectar_low_wav = media_dir / "07_nectar3_harm_lower.wav"
    aux_dbl_wav = media_dir / "08_aux_double.wav"
    aux_sub_wav = media_dir / "09_aux_sub_octave.wav"
    aux_sup_wav = media_dir / "10_aux_super_octave.wav"

    write_pcm24_wav(inst_raw_wav, inst_stereo, target_sr)
    write_pcm24_wav(inst_wav, inst_unmasked, target_sr)
    write_pcm24_wav(main_prep_wav, lead_prepared, target_sr)
    if harm1_mono_for_s1 is not None:
        write_pcm24_wav(harm1_wav, harm1_mono_for_s1, target_sr)
    if harm2_mono_for_s1 is not None:
        write_pcm24_wav(harm2_wav, harm2_mono_for_s1, target_sr)
    has_throw_events = any(ev.fx_type == "ethereal_reverb_throw" for ev in fx_events)
    if has_throw_events:
        write_pcm24_wav(throws_wav, fx_throw_stereo, target_sr)
    write_pcm24_wav(nectar_up_wav, harmony_res.upper_guide_mono, target_sr)
    write_pcm24_wav(nectar_low_wav, harmony_res.lower_guide_mono, target_sr)

    master_wav = out_dir / f"{safe_slug}_AutoMIX_Master.wav"
    vocal_only_wav = out_dir / f"{safe_slug}_AutoMIX_VocalBus.wav"
    write_pcm24_wav(master_wav, final_master_2mix, target_sr)
    vb_pk = float(np.max(np.abs(vocal_bus_stereo))) + 1e-9
    vocal_bus_export = vocal_bus_stereo * (0.89 / vb_pk) if vb_pk > 0.89 else vocal_bus_stereo
    write_pcm24_wav(vocal_only_wav, vocal_bus_export, target_sr)

    print("=== [7/7] Exporting Native Steinberg VST3 Presets & Building Studio One 5 .song Session ===", flush=True)
    patched_presets = export_template_vst3_states(chain_params, bpm=bpm)

    # Mute Nectar 3 tracks by default whenever user already recorded real harmonies (clean soloable guides!)
    mute_nectar = not use_nectar_in_2mix
    nectar_fader_up = -12.5
    nectar_fader_low = -13.2

    extra_tracks: list[TrackMediaSpec] = [
        TrackMediaSpec(name="offvocal_unmasked", wav_path=inst_wav, stereo=True, color="FF2D7DD2", gain_db=0.0, mute=False),
        TrackMediaSpec(name="offvocal_raw_backup", wav_path=inst_raw_wav, stereo=True, color="FF7F8C8D", gain_db=0.0, mute=True),
        TrackMediaSpec(
            name="nectar3_harm_upper",
            wav_path=nectar_up_wav,
            stereo=False,
            color="FF27AE60",
            gain_db=nectar_fader_up,
            mute=mute_nectar,
            pan=0.26,
        ),
        TrackMediaSpec(
            name="nectar3_harm_lower",
            wav_path=nectar_low_wav,
            stereo=False,
            color="FF16A085",
            gain_db=nectar_fader_low,
            mute=mute_nectar,
            pan=0.74,
        ),
    ]

    if dbl_bal is not None:
        write_pcm24_wav(aux_dbl_wav, dbl_bal, target_sr)
        extra_tracks.append(TrackMediaSpec(name="aux_double", wav_path=aux_dbl_wav, stereo=True, color="FFE67E22", gain_db=0.0))
    if sub_oc_bal is not None:
        write_pcm24_wav(aux_sub_wav, sub_oc_bal, target_sr)
        extra_tracks.append(TrackMediaSpec(name="aux_sub_octave", wav_path=aux_sub_wav, stereo=True, color="FF8E44AD", gain_db=0.0))
    if sup_oc_bal is not None:
        write_pcm24_wav(aux_sup_wav, sup_oc_bal, target_sr)
        extra_tracks.append(TrackMediaSpec(name="aux_super_octave", wav_path=aux_sup_wav, stereo=True, color="FF3498DB", gain_db=0.0))

    if has_throw_events and throws_wav.exists():
        extra_tracks.append(
            TrackMediaSpec(
                name="fx_verified_throws",
                wav_path=throws_wav,
                stereo=True,
                color="FF9B51E0",
                gain_db=0.0,
                mute=False,
            )
        )

    combined_markers = list(pt_report.review_markers) + [
        ReviewMarker(time_sec=ev.start_sec, name=f"FX:{ev.fx_type}", category="fx") for ev in fx_events
    ]

    s1_chain_params = type(chain_params)(
        **{**chain_params.to_dict(), "vocal_fader_db": round(chain_params.vocal_fader_db + closed_loop_trim_db, 2)}
    )
    s1_harm_fader_db = round(
        float(np.clip(harmony_target_rel_db + harm_qa_trim_db + s1_chain_params.vocal_fader_db, -18.0, -6.0)), 2
    )

    song_file = build_studio_one_song(
        template_song_path=template_song_path,
        output_song_dir=out_dir,
        song_name=f"{safe_slug}_AutoMIX",
        bpm=bpm,
        sr=target_sr,
        main_vocal_wav=main_prep_wav,
        sections=song_sections,
        extra_tracks=extra_tracks,
        harm1_wav=harm1_wav if harm1_mono_for_s1 is not None else None,
        harm2_wav=harm2_wav if harm2_mono_for_s1 is not None else None,
        patched_presets=patched_presets,
        chain_params=s1_chain_params,
        review_markers=combined_markers,
        harmony_fader_db=s1_harm_fader_db,
    )

    final_full_prof = compute_section_relative_profile(vocal_bus_stereo, inst_unmasked, sr=target_sr)
    final_anchor_bus_prof = compute_section_relative_profile(
        vocal_bus_stereo[:, s0:s1],
        inst_unmasked[:, s0:s1],
        sr=target_sr,
    )

    self_mix_prof_dict = None
    if assets["self_mix"] is not None:
        try:
            sm_sep = BSRoFormerSeparator(
                cache_dir=sep_cache_root / "self_mix_sep",
                model_dir=sep_cache_root / "roformer_models",
            )
            sv_st, si_st, _, _ = sm_sep.separate(
                assets["self_mix"], slug=f"selfmix_{safe_slug}", target_sr=target_sr  # type: ignore[arg-type]
            )
            sa0 = min(s0, max(0, sv_st.shape[1] - target_sr * 10))
            sa1 = min(s1, sv_st.shape[1])
            self_mix_prof_dict = compute_section_relative_profile(
                sv_st[:, sa0:sa1], si_st[:, sa0:sa1], sr=target_sr
            ).to_dict()
        except Exception:
            self_mix_prof_dict = None

    report = {
        "schema_version": "5.0",
        "song_name": song_name,
        "requested_style": style,
        "taste_style": eff_taste,
        "taste_weight": eff_tw,
        "resolved_target_profile": resolved_style,
        "knn_matched_references": knn_neighbors,
        "backing_full_descriptors": full_backing_desc,
        "learned_verse_intimacy_shift_db": learned_verse_shift,
        "detected_key": detected_key_str,
        "bpm": bpm,
        "sample_rate": target_sr,
        "duration_sec": round(common_len / target_sr, 2),
        "studio_one_song": str(song_file),
        "studio_one_injected_presets": sorted(patched_presets.keys()),
        "master_wav": str(master_wav),
        "vocal_bus_wav": str(vocal_only_wav),
        "closed_loop_bus_trim_db": closed_loop_trim_db,
        "harmony_qa_trim_db": harm_qa_trim_db,
        "nectar_mixed_into_2mix": use_nectar_in_2mix,
        "stem_consolidation": {
            "roles": consolidated.stem_roles_summary,
            "deduplicated_skipped": consolidated.deduplicated_files,
            "user_harmony_track_count": consolidated.user_harmony_track_count,
        },
        "stage1_self_calibration": prep_calib.to_dict(),
        "pitch_and_timing": {
            "median_f0_hz": pt_report.median_f0_hz,
            "initial_median_cents_off": pt_report.initial_median_cents_off,
            "already_melodyne_tuned": pt_report.already_melodyne_tuned,
            "pitch_applied": pt_report.pitch_applied,
            "phrases_slipped_count": pt_report.phrases_slipped_count,
            "notes_nudged_count": pt_report.notes_nudged_count,
            "original_align_offset_sec": orig_align_offset_sec,
            "review_markers": [m.to_dict() for m in pt_report.review_markers],
        },
        "nectar3_harmony_summary": {
            "engine_used": harmony_res.engine_used,
            "total_harmony_notes": harmony_res.total_harmony_notes,
            "chord_clash_notes_rescued": harmony_res.chord_clash_notes_rescued,
            "user_recorded_harmony_stems": consolidated.user_harmony_track_count,
            "mixed_into_2mix": use_nectar_in_2mix,
        },
        "optimized_chain_params": s1_chain_params.to_dict(),
        "anchor_section": a_lbl,
        "sections": [
            {
                "index": s.index,
                "label": s.label,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "is_anchor": s.is_anchor,
                "inst_rms_db": s.inst_rms_db,
                "clip_gain_db": s.clip_gain_db,
            }
            for s in song_sections
        ],
        "comparison_anchor_chorus": {
            "pro_target": target_prof.to_dict(),
            "automix_anchor_core": achieved_anchor_prof.to_dict(),
            "automix_full_vocal_bus": final_anchor_bus_prof.to_dict(),
            "user_self_mix": self_mix_prof_dict,
        },
        "automix_full_song_profile": final_full_prof.to_dict(),
        "stage3_creative_fx_events": [ev.to_dict() for ev in fx_events],
        "stage4_director_harmony_plan": [hs.to_dict() for hs in harmony_res.suggestions],
    }

    report_json_path = out_dir / "mix_report.json"
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_director_markdown_report(out_dir / "MIX_DIRECTOR_REPORT.md", report)

    return report


def _write_director_markdown_report(md_path: Path, report: dict) -> None:
    """Write a comprehensive v5 Japanese Director & Psychoacoustic Mix Report."""
    p = report["optimized_chain_params"]
    pt = report["pitch_and_timing"]
    sc = report["stage1_self_calibration"]
    st = report["stem_consolidation"]
    nh = report["nectar3_harmony_summary"]
    fd = report.get("backing_full_descriptors", {})
    knn = report.get("knn_matched_references", [])
    comp = report["comparison_anchor_chorus"]
    tgt = comp["pro_target"]
    bus = comp["automix_full_vocal_bus"]
    slf = comp.get("user_self_mix")

    plugins_joined = " → ".join(p["selected_plugins"])

    def _row(label: str, t_val: str, b_val: str, s_val: str | None) -> str:
        return f"| {label} | `{t_val}` | `{b_val}` | `{s_val}` |" if s_val is not None else f"| {label} | `{t_val}` | `{b_val}` | - |"

    lines = [
        f"# 自動MIX＆ディレクション診断レポート (v5) — {report['song_name']}",
        "",
        f"- **フル尺オケ編成アーキタイプ判定**: `{p['archetype_chain']}` （フル尺密度 `{fd.get('full_inst_mid_density_index', '-')}` / ダイナミックレンジ `{fd.get('backing_dynamic_range_db', '-')} dB`）",
        f"- **適用リファレンス目標プロファイル (44曲音響k-NN＋テイスト)**: `{report['resolved_target_profile']}` （指定: `{report['requested_style']}`）",
        f"- **検出キー / テンポ / 声域中央F0**: `{report['detected_key']}` / `{report['bpm']} BPM` / `{pt['median_f0_hz']} Hz`",
        f"- **Studio One 5 真正Steinberg `.vstpreset` 1対1注入済みチェーン**: `{plugins_joined}`",
        f"- **生成済みStudio One 5 プロジェクト**: `{report['studio_one_song']}` （注入済み真正 `.vstpreset`: `{len(report.get('studio_one_injected_presets', []))} 本`）",
        f"- **完成マスタリング音源**: `{report['master_wav']}`",
        "",
    ]

    if knn:
        lines.extend(["## 0. 44曲プロデータベース音響k-NN近傍マッチング上位", ""])
        for n in knn[:5]:
            lines.append(
                f"- **{n['title']}** (`{n['group']}` / `{n['archetype']}`) — 音響重み `{n['weight_pct']}%` （特徴距離 `{n['distance']}`）"
            )
        lines.append(
            f"- **プロ近傍から学習した平歌・落ちサビ近接シフト**: `+{report.get('learned_verse_intimacy_shift_db', 0.0):.2f} dB`（連続距離トラッカー併用）"
        )
        lines.append("")

    lines.extend(
        [
            "## 1. ステム自動分類・グルーヴ保護＆自己校正下処理",
            "",
            f"- **自動分類されたステム構成**: `{ {k: len(v) for k, v in st['roles'].items()} }` （重複書き出し自動スキップ: `{len(st['deduplicated_skipped'])} ファイル`）",
            f"- **自己校正 Spectral De-ess (`cutoff={sc['deess_cutoff_hz']}Hz`)**: `th={sc['deess_calibrated_threshold_db']:+.1f} dB` → 歯擦音ピーク実測減衰 `{sc['measured_sibilant_gr_db']:.2f} dB` / 1.8〜4.2kHz アタック帯域変化 `{sc['measured_attack_band_delta_db']:+.2f} dB`",
            f"- **ピッチ／タイミング保護判定**: `{'Melodyne補正済み・人間グルーヴ完全保持（無断スリップ無効）' if pt['already_melodyne_tuned'] else '未補正のためPraat PitchTier微補正適用'}` （初期偏差: `{pt['initial_median_cents_off']} ct` / 自動スリップ: `{pt['phrases_slipped_count']}` 件）",
            "",
            "## 2. ハイブリッド最適化VST3設定（微分EQ → 実測GR二分探索 → 連続距離トラッカー）",
            "",
            "| ステージ / プラグイン | パラメータ | 算出値 | 役割 |",
            "|---|---|---|---|",
            f"| **Persistent Q (6-Band)** | HPF / Body(225Hz) / Mud(560Hz) | `{p['hpf_hz']}Hz` / `{p['body_gain_db']:+.2f}dB` / `{p['mud_cut_db']:+.2f}dB` | 固有ボーカル形状の下支えと箱鳴り整理 |",
            f"| **Persistent Q (6-Band)** | Mid(1.25kHz) / Presence(2.9kHz) / Air(10.5kHz) | `{p['core_mid_gain_db']:+.2f}dB` / `{p['presence_gain_db']:+.2f}dB` / `{p['air_shelf_db']:+.2f}dB` | 声の芯・抜け・オケ高域質感との同期 |",
            f"| **Softube VCA Compressor** | Threshold (実測GR) | `{p['vca_threshold_db']} dB` (`GR={p['vca_measured_gr_db']:.2f} dB`) | アタックを残す1段目ピーク制御 |",
            f"| **Stage 2 Leveler** | Calibration Param (Crest GR) | `{p['comp2_param']}` (`-{p['comp2_measured_gr_db']:.2f} dB`) | ボーカル中域密度・位置安定化 |",
            f"| **Softube Saturation Knob** | Amount / Parallel Wet Blend | `{p['parallel_sat_amount']}` / `{p['parallel_sat_wet_pct']}%` | 連続距離連動パラレル倍音付加 |",
            f"| **Bark Adaptive Unmasking** | 14帯域Midスペクトル空間確保 | 最大 `-{p['unmask_mid_dip_db']:.2f} dB` | Side無傷のままオケ中央に声ポケット確保 |",
            f"| **Spatial Glue** | EarlyRoom(8m) / Slap / 1/8 Ducked / Clean Plate | `{p['early_room_send_db']:+.1f}` / `{p['slap_delay_send_db']:+.1f}` / `{p['tempo_delay_send_db']:+.1f}` / `{p['plate_reverb_send_db']:+.1f} dB` | 落ちサビ連動の初期反射＋ディレイ音場接着 |",
            f"| **Harmony Bus QA** | 真ステレオK-LUFS整合＆自動トリム | `Trim {report.get('harmony_qa_trim_db', 0.0):+.2f} dB` | 録音ハモリの歯擦音緩和・室内接着・音量保護 |",
            f"| **Channel Fader** | Main Vocal Fader (QA Trim込) | `{p['vocal_fader_db']:+.2f} dB` | 最終合算バスの聴覚ラウドネスロック |",
            "",
            "## 3. 心理音響バランス＆固有声質比較（基準サビ・全レイヤー合算後）",
            "",
            "| 指標 | 44曲プロk-NN目標 (v5) | 今回のAutoMIX v5 | ご自身の過去MIX |",
            "|---|---|---|---|",
            _row("K特性ラウドネス比 (聴覚総合V/I)", f"{tgt['k_lufs_vi_ratio_db']:+.2f} dB", f"{bus['k_lufs_vi_ratio_db']:+.2f} dB", f"{slf['k_lufs_vi_ratio_db']:+.2f} dB" if slf else None),
            _row("中央Mid K特性ラウドネス比", f"{tgt['mid_k_lufs_vi_ratio_db']:+.2f} dB", f"{bus['mid_k_lufs_vi_ratio_db']:+.2f} dB", f"{slf['mid_k_lufs_vi_ratio_db']:+.2f} dB" if slf else None),
            _row("部分ラウドネス (18帯域マスキング余白)", f"{tgt['partial_loudness_margin_db']:+.2f} dB", f"{bus['partial_loudness_margin_db']:+.2f} dB", f"{slf['partial_loudness_margin_db']:+.2f} dB" if slf else None),
            _row("固有ボーカル芯 140-400Hz (対声コア)", f"{tgt['vocal_intrinsic_bands_db']['body_warmth']:+.2f} dB", f"{bus['vocal_intrinsic_bands_db']['body_warmth']:+.2f} dB", f"{slf['vocal_intrinsic_bands_db']['body_warmth']:+.2f} dB" if slf else None),
            _row("固有ボーカルこもり 400-900Hz", f"{tgt['vocal_intrinsic_bands_db']['boxiness_mud']:+.2f} dB", f"{bus['vocal_intrinsic_bands_db']['boxiness_mud']:+.2f} dB", f"{slf['vocal_intrinsic_bands_db']['boxiness_mud']:+.2f} dB" if slf else None),
            _row("固有ボーカル中域密度 900-1800Hz", f"{tgt['vocal_intrinsic_bands_db']['core_mid']:+.2f} dB", f"{bus['vocal_intrinsic_bands_db']['core_mid']:+.2f} dB", f"{slf['vocal_intrinsic_bands_db']['core_mid']:+.2f} dB" if slf else None),
            _row("固有ボーカル存在感 1.8-4.5kHz", f"{tgt['vocal_intrinsic_bands_db']['core_presence']:+.2f} dB", f"{bus['vocal_intrinsic_bands_db']['core_presence']:+.2f} dB", f"{slf['vocal_intrinsic_bands_db']['core_presence']:+.2f} dB" if slf else None),
            _row("固有ボーカル歯擦音 4.5-8.5kHz", f"{tgt['vocal_intrinsic_bands_db']['sibilance']:+.2f} dB", f"{bus['vocal_intrinsic_bands_db']['sibilance']:+.2f} dB", f"{slf['vocal_intrinsic_bands_db']['sibilance']:+.2f} dB" if slf else None),
            _row("固有ボーカル空気感 8.5-16kHz", f"{tgt['vocal_intrinsic_bands_db']['air_sheen']:+.2f} dB", f"{bus['vocal_intrinsic_bands_db']['air_sheen']:+.2f} dB", f"{slf['vocal_intrinsic_bands_db']['air_sheen']:+.2f} dB" if slf else None),
            _row("歌詞明瞭度指数 (2-8Hz変調深度)", f"{tgt['syllabic_clarity_index']:.3f}", f"{bus['syllabic_clarity_index']:.3f}", f"{slf['syllabic_clarity_index']:.3f}" if slf else None),
            _row("ステレオ広がり (Vocal Side/Mid比)", f"{tgt['vocal_side_mid_db']:+.1f} dB", f"{bus['vocal_side_mid_db']:+.1f} dB", f"{slf['vocal_side_mid_db']:+.1f} dB" if slf else None),
            _row("空間余韻比 (Wet Tail Ratio)", f"{tgt['wet_tail_ratio_db']:+.1f} dB", f"{bus['wet_tail_ratio_db']:+.1f} dB", f"{slf['wet_tail_ratio_db']:+.1f} dB" if slf else None),
            "",
            "## 4. セクション別アンカー伝播（上下ステップ検出＋連続距離トラッカー連動）",
            "",
            "| セクション | 時間区間 | オケRMS | 自動適用クリップゲイン |",
            "|---|---|---|---|",
        ]
    )
    for s in report["sections"]:
        mark = " **(基準アンカー)**" if s["is_anchor"] else ""
        lines.append(
            f"| `{s['label']}`{mark} | `{s['start_sec']:.1f}s - {s['end_sec']:.1f}s` | `{s['inst_rms_db']:.1f} dB` | `{s['clip_gain_db']:+.2f} dB` |"
        )

    lines.extend(["", "## 5. 目標③：本家ボーカル照合による局所特殊演出", ""])
    if report["stage3_creative_fx_events"]:
        for ev in report["stage3_creative_fx_events"]:
            lines.append(f"- **`{ev['start_sec']:.2f}s - {ev['end_sec']:.2f}s`** (`{ev['fx_type']}`): {ev['reason']}")
    else:
        lines.append("- 本家ボーカルに突出したロングリバーブ投げ・ラジオボイス演出は検出されなかったため、不要な勝手飛ばしを行わず自然な空間接着を維持しました。")

    lines.extend(
        [
            "",
            "## 6. 目標④：専用ハモリバス＆iZotope Nectar 3 ガイド運用",
            "",
            f"- **使用エンジン**: `{nh['engine_used']}` （解析ノート数: `{nh['total_harmony_notes']}` / コード衝突回避: `{nh['chord_clash_notes_rescued']} 箇所` / 録音済みハモリ本数: `{nh['user_recorded_harmony_stems']} 本` / 2mix合算: `{'ON（録音ハモリなしのため自動採用）' if nh.get('mixed_into_2mix') else 'OFF（録音ハモリ優先・Studio One上ではミュートガイド配置）'}`）",
            "",
        ]
    )
    for hs in report["stage4_director_harmony_plan"]:
        layers_str = " / ".join(hs["recommended_layers"])
        lines.append(
            f"- **`{hs['section_label']}` (`{hs['start_sec']:.1f}s - {hs['end_sec']:.1f}s` — {hs['arrangement_role']})**: `[{layers_str}]` — {hs['director_note_ja']}"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def repropagate_anchor_in_song(
    song_path: Path,
    anchor_event_index: int = 0,
) -> Path:
    """Read a user-tweaked Studio One 5 `.song`, take their adjusted anchor clip gain delta, and propagate to all sections!"""
    with zipfile.ZipFile(song_path, "r") as zin:
        files = {n: zin.read(n) for n in zin.namelist()}

    xml_text = files["Song/song.xml"].decode("utf-8")
    parseable_xml = xml_text.lstrip("\ufeff").replace(
        "<Song ", '<Song xmlns:x="http://www.presonus.com/xml" ', 1
    )
    root = ET.fromstring(parseable_xml)

    levels: list[float] = []
    for mtrack in root.iter("MediaTrack"):
        if mtrack.attrib.get("name") == "mainvocal":
            for ev in mtrack.iter("AudioEvent"):
                vc = ev.find("./*[@name='volumeCurve']") or ev.find("Attributes")
                lv = float(vc.attrib.get("level", "1.0")) if vc is not None else 1.0
                levels.append(lv)

    if not levels or anchor_event_index >= len(levels):
        raise ValueError(f"Could not find event index {anchor_event_index} on mainvocal track.")

    report_path = song_path.parent / "mix_report.json"
    orig_sections = json.loads(report_path.read_text(encoding="utf-8"))["sections"] if report_path.exists() else None
    orig_anchor_db = orig_sections[anchor_event_index]["clip_gain_db"] if orig_sections else 0.0
    new_anchor_db = 20.0 * np.log10(max(levels[anchor_event_index], 1e-4))
    user_delta_db = new_anchor_db - orig_anchor_db

    out_path = song_path.with_name(song_path.stem + "_propagated.song")
    updated_xml = xml_text
    if orig_sections:
        for s in orig_sections:
            ev_name = f"main_{s['label']}"
            new_db = s["clip_gain_db"] + (0.0 if "Instrumental" in s["label"] else user_delta_db)
            new_lin = round(float(10.0 ** (new_db / 20.0)), 6)
            marker = f'name="{ev_name}"'
            idx_m = updated_xml.find(marker)
            if idx_m != -1:
                idx_lvl = updated_xml.find('level="', idx_m)
                idx_end = updated_xml.find('"', idx_lvl + 7)
                if idx_lvl != -1 and idx_end != -1 and (idx_lvl - idx_m) < 260:
                    updated_xml = updated_xml[: idx_lvl + 7] + f"{new_lin:.8f}" + updated_xml[idx_end:]

    files["Song/song.xml"] = updated_xml.encode("utf-8")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for k, v in files.items():
            zout.writestr(k, v)

    return out_path
