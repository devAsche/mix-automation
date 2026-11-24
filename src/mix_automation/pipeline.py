"""v2 End-to-End Automated Vocal Mixing, Nectar 3 Harmony & Studio One 5 Session Pipeline.

Integrates all v2 architectural upgrades into a deterministic, closed-loop production system:
  1. Separates `original_*` reference track via GPU `BS-RoFormer` and time-aligns it to the
     distributed off-vocal so slip-timing & Stage 3 special FX cues are 100% evidence-backed.
  2. Runs clean Stage 1 pre-processing (`RX 10` VST3s, Melodyne-aware conditional `TD-PSOLA`,
     vowel-core gain riding, genuine `RX 10 De-ess` + `Soothe2`, and `-18.0 LUFS` staging).
  3. Automatically stratifies the target profile by singer vocal register (`low_mid` vs `high`)
     and backing track archetype (`dense_rock_edm`, `groovy_mid_pop`, `intimate_ballad_sparse`),
     optimizing genuine commercial VST3 chains + delay-first spatial blend end-to-end via CMA-ES.
  4. Synthesizes chord- & modulation-safe harmonies via `iZotope Nectar 3.vst3` that function
     both as turn-key mix layers and soloable DAW recording guides.
  5. Enforces Closed-Loop Post-Sum Vocal Bus QA (verifying total summed vocal-to-inst ratio,
     wet-tail ratio, and syllabic clarity after all harmonies/throws are added) followed by
     cohesive 2-mix VCA bus glue and safe bounded true-peak mastering (`<= 3.2 dB` limiter drive).
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

from mix_automation.analysis.reference_extractor import BSRoFormerSeparator
from mix_automation.analysis.relative_profile import (
    SectionRelativeProfile,
    compute_anchor_propagation,
    compute_section_relative_profile,
)
from mix_automation.creative.fx_and_harmony import (
    apply_verified_radio_fx_inline,
    detect_creative_fx_events,
    generate_nectar3_harmonies_and_plan,
    render_ethereal_throw_stem,
)
from mix_automation.dsp.pitch_timing import ReviewMarker
from mix_automation.dsp.preprocess import PreprocessConfig, preprocess_vocal_pipeline
from mix_automation.dsp.vst3_optimizer import (
    SongSection,
    apply_2mix_bus_glue_and_safe_master,
    apply_dynamic_instrumental_unmasking,
    detect_song_sections,
    optimize_real_vst3_chain,
)
from mix_automation.io.audio_loader import consolidate_vocal_stem_folder, load_audio
from mix_automation.io.song_builder import (
    TrackMediaSpec,
    build_studio_one_song,
    export_template_vst3_states,
    write_pcm24_wav,
)


def load_target_profile(
    profiles_json: Path,
    style: str = "auto",
    vocal_median_f0_hz: float = 260.0,
    backing_archetype: str = "dense_rock_edm",
) -> tuple[SectionRelativeProfile, str]:
    """Load the stratified or explicit engineer target profile (`schema_version` 2.0 compatible)."""
    data = json.loads(profiles_json.read_text(encoding="utf-8"))
    targets = data["target_profiles"]
    req = style.lower().strip()

    reg_key = "low_mid" if vocal_median_f0_hz < 295.0 else "high"
    stratified_key = f"{reg_key}_{backing_archetype}"
    reg_fallback_key = f"register_{reg_key}"

    if req == "auto":
        bucket_n = sum(
            1
            for s in data.get("songs", [])
            if s.get("vocal_register") == reg_key and s.get("backing_archetype") == backing_archetype
        )
        if stratified_key in targets and (bucket_n >= 3 or not data.get("songs")):
            resolved = stratified_key
        elif reg_fallback_key in targets:
            resolved = reg_fallback_key
        else:
            resolved = "consensus"
    else:
        if req not in targets:
            raise ValueError(f"Style {style!r} not found. Available: {['auto'] + list(targets.keys())}")
        resolved = req

    d = targets[resolved]
    prof = SectionRelativeProfile(
        vocal_rms_db=d["vocal_rms_db"],
        inst_rms_db=d["inst_rms_db"],
        broadband_vi_ratio_db=d["broadband_vi_ratio_db"],
        vocal_crest_db=d["vocal_crest_db"],
        inst_crest_db=d["inst_crest_db"],
        crest_difference_db=d["crest_difference_db"],
        position_stability_std_db=d["position_stability_std_db"],
        harsh_resonance_index_db=d["harsh_resonance_index_db"],
        syllabic_clarity_index=d.get("syllabic_clarity_index", 0.36),
        wet_tail_ratio_db=d.get("wet_tail_ratio_db", -21.5),
        vocal_side_mid_db=d.get("vocal_side_mid_db", -11.0),
        backing_archetype=d.get("backing_archetype", resolved),
        band_margins_db=d["band_margins_db"],
        inst_band_levels_db=d["inst_band_levels_db"],
        active_frame_ratio=d["active_frame_ratio"],
    )
    return prof, resolved


def discover_song_assets(song_folder: Path) -> dict[str, Path | list[Path] | None]:
    """Automatically identify stems, off-vocal, original reference, lyrics, and self-mix."""
    stems_dir = song_folder / "stems"
    stem_files = sorted(stems_dir.glob("*.wav")) if stems_dir.exists() else []

    inst_file: Path | None = None
    orig_file: Path | None = None
    self_mix_file: Path | None = None
    lyrics_file: Path | None = None

    for f in sorted(song_folder.iterdir()):
        if f.is_dir():
            continue
        low = f.name.lower()
        if f.suffix.lower() == ".txt":
            lyrics_file = f
        elif "自mix" in low and "オケなし" not in low:
            self_mix_file = f
        elif any(k in low for k in ("inst", "off vocal", "offvocal", "off_vocal", "オケ")) and "自mix" not in low:
            inst_file = f
        elif f.suffix.lower() in {".mp3", ".wav", ".webm", ".flac", ".m4a"} and "自mix" not in low:
            orig_file = f

    if inst_file is None:
        raise FileNotFoundError(f"Could not auto-detect off-vocal/INST file in {song_folder}")
    if not stem_files:
        raise FileNotFoundError(f"No vocal stem WAVs found inside {stems_dir}")

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
    hop = int(sr * 0.01)  # 10ms envelope frames
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


def run_full_automix_for_song(
    song_folder: Path,
    template_song_path: Path,
    profiles_json_path: Path,
    output_root: Path,
    style: str = "auto",
    target_sr: int = 44100,
    force_auto_pitch: bool = False,
) -> dict:
    """Execute the complete v2 automated vocal mixing & Studio One 5 session pipeline."""
    song_name = song_folder.name
    safe_slug = song_name.replace("&", "and").replace(" ", "_")
    out_dir = output_root / f"{safe_slug}_{style}"
    media_dir = out_dir / "Media"
    media_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [1/7] Loading stems & separating original reference for '{song_name}' ===", flush=True)
    assets = discover_song_assets(song_folder)
    inst_stereo, _ = load_audio(assets["inst"], target_sr=target_sr, mono=False)  # type: ignore[arg-type]
    inst_mono = np.mean(inst_stereo, axis=0, dtype=np.float32)

    consolidated = consolidate_vocal_stem_folder(assets["stems"], target_sr=target_sr)  # type: ignore[arg-type]

    common_len = max(inst_stereo.shape[1], len(consolidated.lead_mono))
    if inst_stereo.shape[1] < common_len:
        inst_stereo = np.pad(inst_stereo, ((0, 0), (0, common_len - inst_stereo.shape[1])))
        inst_mono = np.pad(inst_mono, (0, common_len - len(inst_mono)))
    lead_raw = np.pad(consolidated.lead_mono, (0, max(0, common_len - len(consolidated.lead_mono))))[:common_len]

    tempo_est, _ = librosa.beat.beat_track(y=inst_mono[: min(len(inst_mono), target_sr * 90)], sr=target_sr)
    bpm = round(float(np.atleast_1d(tempo_est)[0]), 1)
    if bpm < 75.0:
        bpm *= 2.0

    # Separate original reference song via GPU BS-RoFormer & time-align to off-vocal
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

    print("=== [2/7] Stage 1 Clean Pre-processing (RX 10 + Smart Slip/TD-PSOLA + Rider + Soothe2 + -18 LUFS) ===", flush=True)
    prep_cfg = PreprocessConfig(force_auto_pitch=force_auto_pitch)
    lead_prepared, pt_report, scale_info = preprocess_vocal_pipeline(
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
        f"Key: {detected_key_str} | Register F0: {pt_report.median_f0_hz}Hz | "
        f"Melodyne Pre-Tuned: {pt_report.already_melodyne_tuned} (median {pt_report.initial_median_cents_off}ct) | "
        f"Slipped Phrases: {pt_report.phrases_slipped_count} | PSOLA Nudged: {pt_report.notes_nudged_count}",
        flush=True,
    )

    print(f"=== [3/7] Archetype Selection & Real-VST3 End-to-End CMA-ES Optimization (Style: '{style}') ===", flush=True)
    raw_spans, anchor_idx = detect_song_sections(lead_prepared, inst_mono, sr=target_sr)
    a_start, a_end, a_lbl = raw_spans[anchor_idx]
    s0, s1 = int(a_start * target_sr), int(a_end * target_sr)

    init_anchor_prof = compute_section_relative_profile(lead_prepared[s0:s1], inst_stereo[:, s0:s1], sr=target_sr)
    target_prof, resolved_style = load_target_profile(
        profiles_json_path,
        style=style,
        vocal_median_f0_hz=pt_report.median_f0_hz,
        backing_archetype=init_anchor_prof.backing_archetype,
    )

    rig, chain_params, achieved_anchor_prof, final_loss = optimize_real_vst3_chain(
        lead_prepared[s0:s1],
        inst_stereo[:, s0:s1],
        sr=target_sr,
        bpm=bpm,
        target_profile=target_prof,
        n_iterations=32,
    )
    print(
        f"Archetype: '{chain_params.archetype_chain}' | Resolved Profile: '{resolved_style}' | "
        f"Anchor '{a_lbl}' (loss={final_loss}) -> HPF={chain_params.hpf_hz}Hz, Body={chain_params.body_gain_db:+.2f}dB, "
        f"Mud={chain_params.mud_cut_db:+.2f}dB, Presence={chain_params.presence_gain_db:+.2f}dB, "
        f"Air={chain_params.air_shelf_db:+.2f}dB, Fader={chain_params.vocal_fader_db:+.2f}dB, "
        f"Slap={chain_params.slap_delay_send_db:+.1f}dB, Plate={chain_params.plate_reverb_send_db:+.1f}dB",
        flush=True,
    )

    print("=== [4/7] Full-Song Real VST3 Rendering & Section Anchor Propagation ===", flush=True)
    lead_chained_base = rig.apply_insert_chain(lead_prepared, sr=target_sr, p=chain_params)

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
            cg_db = round(float(np.clip(adj.input_gain_delta_db * 0.65, -3.5, 3.0)), 2)
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

    xf_win = np.hanning(int(0.12 * target_sr)).astype(np.float32)
    xf_win /= np.sum(xf_win)
    section_envelope = signal.fftconvolve(section_envelope, xf_win, mode="same").astype(np.float32)
    lead_final_dry = (lead_chained_base * section_envelope).astype(np.float32)

    print("=== [5/7] Stage 3 Reference-Verified FX & Stage 4 Nectar 3 Chord-Safe Harmonies ===", flush=True)
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
        f"({harmony_res.total_harmony_notes} notes, {harmony_res.chord_clash_notes_rescued} chord clashes rescued)",
        flush=True,
    )

    print("=== [6/7] Mixing Supporting Layers, Closed-Loop Bus QA & Safe 2-Mix Bus Glue ===", flush=True)
    spatial_wet_stereo = rig.render_spatial_blend(lead_final_dry, sr=target_sr, p=chain_params)

    lead_active = lead_final_dry[np.abs(lead_final_dry) > 1e-4]
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
                PeakFilter(cutoff_frequency_hz=2850.0, gain_db=-2.2, q=1.0),
                LowpassFilter(9800.0),
                Compressor(threshold_db=-22.0, ratio=3.8, attack_ms=8.0, release_ms=110.0),
            ]
        )
        proc = b(padded, target_sr)
        cur_rms = float(np.sqrt(np.mean(proc[np.abs(proc) > 1e-4] ** 2) + 1e-12))
        target_rms = lead_active_rms * (10.0 ** (rel_db / 20.0))
        gain = min(target_rms / cur_rms, 3.2)
        return (proc * gain).astype(np.float32)

    aux_layers_stereo = np.zeros((2, common_len), dtype=np.float32)
    harm1_mono_for_s1: np.ndarray | None = None
    harm2_mono_for_s1: np.ndarray | None = None

    if consolidated.double_stereo is not None:
        dbl_bal = _balance_aux_stereo(consolidated.double_stereo, rel_db=-10.5, hpf=195.0)
        if dbl_bal is not None:
            aux_layers_stereo += dbl_bal

    for oc_st, rel_db, hpf_hz in [
        (consolidated.sub_octave_stereo, -13.5, 105.0),
        (consolidated.super_octave_stereo, -14.0, 260.0),
    ]:
        oc_bal = _balance_aux_stereo(oc_st, rel_db=rel_db, hpf=hpf_hz)
        if oc_bal is not None:
            aux_layers_stereo += oc_bal

    if consolidated.harmonies_stereo_bus is not None:
        h_bal = _balance_aux_stereo(consolidated.harmonies_stereo_bus, rel_db=-9.2, hpf=200.0)
        if h_bal is not None:
            aux_layers_stereo += h_bal
        if consolidated.harm1_mono is not None:
            harm1_mono_for_s1 = np.pad(
                consolidated.harm1_mono, (0, max(0, common_len - len(consolidated.harm1_mono)))
            )[:common_len]
        if consolidated.harm2_mono is not None:
            harm2_mono_for_s1 = np.pad(
                consolidated.harm2_mono, (0, max(0, common_len - len(consolidated.harm2_mono)))
            )[:common_len]

    # Blend chord-safe Nectar 3 stereo harmonies (automatically ducked where user harmonies sing!)
    aux_layers_stereo += harmony_res.auto_mix_harmony_stereo

    inst_unmasked = apply_dynamic_instrumental_unmasking(
        inst_stereo, lead_final_dry, sr=target_sr, max_dip_db=chain_params.unmask_mid_dip_db
    )

    # --- Closed-Loop Post-Sum Vocal Bus QA ---
    wet_total = spatial_wet_stereo + fx_throw_stereo
    vocal_bus_stereo = np.stack([lead_final_dry, lead_final_dry], axis=0) + aux_layers_stereo + wet_total

    pre_qa_anchor = compute_section_relative_profile(
        vocal_bus_stereo[:, s0:s1], inst_unmasked[:, s0:s1], sr=target_sr
    )
    # 1. Trim wet tail if total wet space exceeds pro clarity ceiling
    wet_ceiling_db = max(target_prof.wet_tail_ratio_db + 1.8, -18.5)
    if pre_qa_anchor.wet_tail_ratio_db > wet_ceiling_db:
        wet_cut_db = max(-8.0, wet_ceiling_db - pre_qa_anchor.wet_tail_ratio_db)
        wet_total = wet_total * (10.0 ** (wet_cut_db / 20.0))
        vocal_bus_stereo = np.stack([lead_final_dry, lead_final_dry], axis=0) + aux_layers_stereo + wet_total
        pre_qa_anchor = compute_section_relative_profile(
            vocal_bus_stereo[:, s0:s1], inst_unmasked[:, s0:s1], sr=target_sr
        )

    # 2. Trim overall vocal bus broadband gain so final summed bus sits IN the mix, never ON TOP!
    vi_overshoot_db = pre_qa_anchor.broadband_vi_ratio_db - target_prof.broadband_vi_ratio_db
    closed_loop_trim_db = 0.0
    if abs(vi_overshoot_db) > 0.55:
        closed_loop_trim_db = round(float(np.clip(-vi_overshoot_db * 0.85, -4.5, 2.5)), 2)
        vocal_bus_stereo = (vocal_bus_stereo * (10.0 ** (closed_loop_trim_db / 20.0))).astype(np.float32)

    final_master_2mix = apply_2mix_bus_glue_and_safe_master(
        inst_unmasked, vocal_bus_stereo, sr=target_sr, target_lufs=-9.0
    )

    # Write Media WAVs for Studio One 5 session
    inst_wav = media_dir / "01_offvocal_unmasked.wav"
    main_prep_wav = media_dir / "02_mainvocal_prepared.wav"
    harm1_wav = media_dir / "03_harmvocal_1.wav"
    harm2_wav = media_dir / "04_harmvocal_2.wav"
    throws_wav = media_dir / "05_fx_verified_throws.wav"
    nectar_up_wav = media_dir / "06_nectar3_harm_upper.wav"
    nectar_low_wav = media_dir / "07_nectar3_harm_lower.wav"

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

    print("=== [7/7] Exporting Clean VST3 Presets & Building Studio One 5 .song Session ===", flush=True)
    patched_presets = export_template_vst3_states(chain_params, bpm=bpm)

    has_user_harms = consolidated.has_user_harmonies
    nectar_fader_up = -14.5 if has_user_harms else -10.8
    nectar_fader_low = -15.2 if has_user_harms else -11.5

    extra_tracks: list[TrackMediaSpec] = [
        TrackMediaSpec(name="offvocal", wav_path=inst_wav, stereo=True, color="FF2D7DD2", gain_db=0.0, mute=False),
        TrackMediaSpec(
            name="nectar3_harm_upper",
            wav_path=nectar_up_wav,
            stereo=False,
            color="FF27AE60",
            gain_db=nectar_fader_up,
            mute=False,
            pan=0.28,
        ),
        TrackMediaSpec(
            name="nectar3_harm_lower",
            wav_path=nectar_low_wav,
            stereo=False,
            color="FF16A085",
            gain_db=nectar_fader_low,
            mute=False,
            pan=0.72,
        ),
    ]
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

    # Apply closed_loop_trim_db onto Studio One's mainvocal fader too so DAW playback matches Master WAV!
    s1_chain_params = type(chain_params)(
        **{**chain_params.to_dict(), "vocal_fader_db": round(chain_params.vocal_fader_db + closed_loop_trim_db, 2)}
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
        "schema_version": "2.0",
        "song_name": song_name,
        "requested_style": style,
        "resolved_target_profile": resolved_style,
        "detected_key": detected_key_str,
        "bpm": bpm,
        "sample_rate": target_sr,
        "duration_sec": round(common_len / target_sr, 2),
        "studio_one_song": str(song_file),
        "master_wav": str(master_wav),
        "vocal_bus_wav": str(vocal_only_wav),
        "closed_loop_bus_trim_db": closed_loop_trim_db,
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
            "user_recorded_harmony_stems": 2 if consolidated.has_user_harmonies else 0,
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
    """Write a comprehensive v2 Japanese Director & Mix Settings Report."""
    p = report["optimized_chain_params"]
    pt = report["pitch_and_timing"]
    nh = report["nectar3_harmony_summary"]
    comp = report["comparison_anchor_chorus"]
    tgt = comp["pro_target"]
    bus = comp["automix_full_vocal_bus"]
    slf = comp.get("user_self_mix")

    plugins_joined = " → ".join(p["selected_plugins"])

    lines = [
        f"# 自動MIX＆ディレクション診断レポート (v2) — {report['song_name']}",
        "",
        f"- **自動判定オケ編成アーキタイプ**: `{p['archetype_chain']}`",
        f"- **適用リファレンス目標プロファイル**: `{report['resolved_target_profile']}` （指定: `{report['requested_style']}`）",
        f"- **検出キー / テンポ / 声域中央F0**: `{report['detected_key']}` / `{report['bpm']} BPM` / `{pt['median_f0_hz']} Hz`",
        f"- **自動選択された商用VST3チェーン**: `{plugins_joined}`",
        f"- **生成済みStudio One 5 プロジェクト**: `{report['studio_one_song']}`",
        f"- **完成マスタリング音源**: `{report['master_wav']}`",
        "",
        "## 1. 下処理・ピッチ＆タイミング診断（ケロケロ防止＆スリップ補正）",
        "",
        f"- **Melodyne補正済み判定**: `{'補正済みのため再ボコーダ処理を自動スキップ（音質劣化ゼロ）' if pt['already_melodyne_tuned'] else '未補正のためPraat TD-PSOLAで持続音のみ自然補正'}` （初期ピッチ中央偏差: `{pt['initial_median_cents_off']} cents`）",
        f"- **無伸縮フレーズ・スリップタイミング補正**: `{pt['phrases_slipped_count']} フレーズ` （本家ボーカル同期オフセット: `{pt['original_align_offset_sec']:+.2f}s`）",
        f"- **Studio One タイムライン確認マーカー出力数**: `{len(pt['review_markers'])} 箇所`",
        "",
        "## 2. 最適化されたVST3＆空間ブレンド設定値（ディレイ主軸・クリーンPlate設計）",
        "",
        "| ステージ / 処理 | パラメータ | 算出値 | 設計意図 |",
        "|---|---|---|---|",
        f"| **Persistent Q (Dynamic)** | Band 1 Highpass (18dB/oct) | `{p['hpf_hz']} Hz` | チェストの太さを残す安全なローカット |",
        f"| **Persistent Q (Dynamic)** | Band 2 Body (235 Hz, Dyn -2.5dB) | `{p['body_gain_db']:+.2f} dB` | 低音域の膨らみを動的制御しつつ芯を維持 |",
        f"| **Persistent Q** | Band 3 Mud (540 Hz, Q=1.35) | `{p['mud_cut_db']:+.2f} dB` | こもり・箱鳴りの整理 |",
        f"| **Persistent Q (Dynamic)** | Band 4 Presence (2850 Hz, Dyn -2.2dB) | `{p['presence_gain_db']:+.2f} dB` | 高音ベルト時の刺さりを抑えつつ前へ配置 |",
        f"| **Persistent Q** | Band 5 Air Shelf (10.5 kHz) | `{p['air_shelf_db']:+.2f} dB` | シルキーな空気感 |",
        f"| **Dynamics & Color** | Comp1 Drive / Comp2 Glue / Warmth | `{p['comp1_drive']:.2f}` / `{p['comp2_glue']:.2f}` / `{p['color_warmth']:.2f}` | アーキタイプ別の2段コンプ＆倍音質感 |",
        f"| **Delay-First Spatial** | Slap(78/92ms) / Ducked 1/8 Delay / Clean Plate | `{p['slap_delay_send_db']:+.1f}` / `{p['tempo_delay_send_db']:+.1f}` / `{p['plate_reverb_send_db']:+.1f} dB` | ボワン感を排除しディレイと無変調短Plateでオケに接着 |",
        f"| **Channel Fader** | Main Vocal Fader (Closed-Loop Trim込) | `{p['vocal_fader_db']:+.2f} dB` | ハモリ合算後の最終バス比率がプロ目標に一致するよう自動トリム |",
        "",
        "## 3. 対オケ相対バランス＆明瞭度比較（基準サビ・全レイヤー合算後）",
        "",
        "| 指標 | 35曲プロ目標 | 今回のAutoMIX (最終VocalBus) | ご自身の過去MIX |",
        "|---|---|---|---|",
        f"| 対オケ総合ラウドネス比 | `{tgt['broadband_vi_ratio_db']:+.2f} dB` | `{bus['broadband_vi_ratio_db']:+.2f} dB` | `{slf['broadband_vi_ratio_db']:+.2f} dB` |" if slf else f"| 対オケ総合ラウドネス比 | `{tgt['broadband_vi_ratio_db']:+.2f} dB` | `{bus['broadband_vi_ratio_db']:+.2f} dB` | - |",
        f"| 芯の太さ (160-420Hz) | `{tgt['band_margins_db']['body_warmth']:+.2f} dB` | `{bus['band_margins_db']['body_warmth']:+.2f} dB` | `{slf['band_margins_db']['body_warmth']:+.2f} dB` |" if slf else f"| 芯の太さ (160-420Hz) | `{tgt['band_margins_db']['body_warmth']:+.2f} dB` | `{bus['band_margins_db']['body_warmth']:+.2f} dB` | - |",
        f"| 張り付き存在感 (1.5-4.2kHz) | `{tgt['band_margins_db']['core_presence']:+.2f} dB` | `{bus['band_margins_db']['core_presence']:+.2f} dB` | `{slf['band_margins_db']['core_presence']:+.2f} dB` |" if slf else f"| 張り付き存在感 (1.5-4.2kHz) | `{tgt['band_margins_db']['core_presence']:+.2f} dB` | `{bus['band_margins_db']['core_presence']:+.2f} dB` | - |",
        f"| 歌詞明瞭度指数 (高いほどハッキリ) | `{tgt['syllabic_clarity_index']:.3f}` | `{bus['syllabic_clarity_index']:.3f}` | `{slf['syllabic_clarity_index']:.3f}` |" if slf else f"| 歌詞明瞭度指数 (高いほどハッキリ) | `{tgt['syllabic_clarity_index']:.3f}` | `{bus['syllabic_clarity_index']:.3f}` | - |",
        f"| 空間余韻比 (低すぎず高すぎない値) | `{tgt['wet_tail_ratio_db']:+.1f} dB` | `{bus['wet_tail_ratio_db']:+.1f} dB` | `{slf['wet_tail_ratio_db']:+.1f} dB` |" if slf else f"| 空間余韻比 (低すぎず高すぎない値) | `{tgt['wet_tail_ratio_db']:+.1f} dB` | `{bus['wet_tail_ratio_db']:+.1f} dB` | - |",
        "",
        "## 4. セクション別アンカー伝播（パート間音圧追従クリップゲイン）",
        "",
        "| セクション | 時間区間 | オケRMS | 自動適用クリップゲイン |",
        "|---|---|---|---|",
    ]
    for s in report["sections"]:
        mark = " **(基準アンカー)**" if s["is_anchor"] else ""
        lines.append(
            f"| `{s['label']}`{mark} | `{s['start_sec']:.1f}s - {s['end_sec']:.1f}s` | `{s['inst_rms_db']:.1f} dB` | `{s['clip_gain_db']:+.2f} dB` |"
        )

    lines.extend(["", "## 5. 目標③：本家ボーカル比較で裏付けられた局所特殊演出", ""])
    if report["stage3_creative_fx_events"]:
        for ev in report["stage3_creative_fx_events"]:
            lines.append(f"- **`{ev['start_sec']:.2f}s - {ev['end_sec']:.2f}s`** (`{ev['fx_type']}`): {ev['reason']}")
    else:
        lines.append("- 本家ボーカルに突出したロングリバーブ投げ・ラジオボイス演出は検出されなかったため、不要な勝手飛ばしを行わずタイトな空間ブレンドを維持しました。")

    lines.extend(
        [
            "",
            "## 6. 目標④：iZotope Nectar 3 コード追従ハモリ生成＆ディレクション",
            "",
            f"- **使用エンジン**: `{nh['engine_used']}` （解析ノート数: `{nh['total_harmony_notes']}` / スケール外・借用和音の不協和を自動回避したノート数: `{nh['chord_clash_notes_rescued']} 箇所`）",
            "- **完パケ＆ガイド両対応設計**: Studio One 5 内の `nectar3_harm_upper` (Left) / `nectar3_harm_lower` (Right) トラックはそのまま再生すれば完パケ用の馴染む音量バランスで鳴り、ソロ（S）ボタンを押せばご自身の声質での録音用ガイドボーカルとしてクリアに聴けます。",
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
