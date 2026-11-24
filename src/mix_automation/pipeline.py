"""End-to-End Automated Vocal Mixing & Anchor Re-propagation Pipeline.

Integrates Stages ①, ②, ③, and ④ into a single deterministic engineering workflow:
  1. Auto-discovers raw vocal stems, backing track, lyrics, original song, and self-mix reference
  2. Runs Stage 1 iZotope RX 10 cleanup + nuance-safe pitch centering + phoneme-aware gain riding
  3. Segments song structure, optimizes the vocal chain against the 15-song pro target profile
     (`consensus`, `azusagawa`, `toy`, or `saburou`) via CMA-ES, and propagates section clip gains
  4. Applies real Oeksound Soothe2 & RX 10 De-ess polish, spatial R2/Plate reverb & H-Delay sends,
     dynamic instrumental unmasking, Stage 3 ethereal phrase-end throws, Stage 4 harmony guides,
     and true-peak mastering (`Wave Breaker` / loudness normalization to -8.2 LUFS)
  5. Outputs a complete Studio One 5.5 `.song` session folder, rendered 2mix & vocal stems, and
     a detailed Markdown & JSON Director Report comparing Raw / Self-Mix / Auto-MIX / Pro Target!
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
from pedalboard import (
    Compressor,
    Delay,
    HighpassFilter,
    Limiter,
    LowpassFilter,
    Pedalboard,
    Reverb,
    load_plugin,
)
from scipy import signal

from mix_automation.analysis.relative_profile import (
    SectionRelativeProfile,
    compute_anchor_propagation,
    compute_section_relative_profile,
)
from mix_automation.config.plugin_registry import get_plugin_map
from mix_automation.creative.fx_and_harmony import (
    detect_creative_fx_events,
    generate_harmony_guides_and_plan,
    render_ethereal_throw_stem,
)
from mix_automation.dsp.preprocess import PreprocessConfig, preprocess_vocal_pipeline
from mix_automation.dsp.vst3_optimizer import (
    SongSection,
    VocalChainParams,
    apply_dynamic_instrumental_unmasking,
    apply_fast_parametric_vocal_chain,
    detect_song_sections,
    optimize_vocal_chain_to_target,
    render_real_vst3_Polish_pass,
)
from mix_automation.io.audio_loader import consolidate_vocal_stem_folder, load_audio
from mix_automation.io.song_builder import (
    TrackMediaSpec,
    build_studio_one_song,
    export_template_vst3_states,
    write_pcm24_wav,
)


def load_target_profile(profiles_json: Path, style: str = "consensus") -> SectionRelativeProfile:
    """Load one of the pre-computed pro target profiles (`consensus`, `azusagawa`, `toy`, `saburou`)."""
    data = json.loads(profiles_json.read_text(encoding="utf-8"))
    targets = data["target_profiles"]
    key = style.lower()
    if key not in targets:
        raise ValueError(f"Style {style!r} not found. Available: {list(targets.keys())}")
    d = targets[key]
    return SectionRelativeProfile(
        vocal_rms_db=d["vocal_rms_db"],
        inst_rms_db=d["inst_rms_db"],
        broadband_vi_ratio_db=d["broadband_vi_ratio_db"],
        vocal_crest_db=d["vocal_crest_db"],
        inst_crest_db=d["inst_crest_db"],
        crest_difference_db=d["crest_difference_db"],
        position_stability_std_db=d["position_stability_std_db"],
        harsh_resonance_index_db=d["harsh_resonance_index_db"],
        band_margins_db=d["band_margins_db"],
        inst_band_levels_db=d["inst_band_levels_db"],
        active_frame_ratio=d["active_frame_ratio"],
    )


def discover_song_assets(song_folder: Path) -> dict[str, Path | list[Path] | None]:
    """Automatically identify stems, off-vocal, original reference, lyrics, and self-mix in a raw song folder."""
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
        elif f.suffix.lower() in {".mp3", ".wav", ".webm", ".flac"} and "自mix" not in low:
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


def _render_spatial_sends(
    lead_vocal_mono: np.ndarray,
    sr: int,
    params: VocalChainParams,
    bpm: float,
) -> np.ndarray:
    """Render lush stereo reverb (`R2` / `Rev PLATE-140`) + tempo-synced ping-pong delay sends."""
    pmap = get_plugin_map()
    st_in = np.stack([lead_vocal_mono, lead_vocal_mono], axis=0).astype(np.float32)

    # 1. Reverb send
    rev_wet: np.ndarray | None = None
    if pmap["exponential_r2"].exists:
        try:
            r2 = load_plugin(str(pmap["exponential_r2"].loadable_path))
            if "mix" in r2.parameters:
                r2.mix = 100.0
            rev_board = Pedalboard([HighpassFilter(220.0), LowpassFilter(9500.0), r2])
            rev_wet = rev_board(st_in, sr)
        except Exception:
            rev_wet = None

    if rev_wet is None:
        rev_board = Pedalboard(
            [
                HighpassFilter(220.0),
                LowpassFilter(9500.0),
                Reverb(room_size=0.62, damping=0.48, wet_level=1.0, dry_level=0.0, width=0.95),
            ]
        )
        rev_wet = rev_board(st_in, sr)

    # 2. Tempo-synced filtered stereo delay send (eighth/quarter note feel)
    delay_sec = float(np.clip(60.0 / max(bpm, 60.0) * 0.5, 0.16, 0.45))
    dly_board = Pedalboard(
        [
            HighpassFilter(350.0),
            LowpassFilter(4200.0),
            Delay(delay_seconds=delay_sec, feedback=0.24, mix=1.0),
        ]
    )
    dly_wet = dly_board(st_in, sr)

    rev_gain = 10.0 ** (params.reverb_send_db / 20.0)
    dly_gain = 10.0 ** (params.delay_send_db / 20.0)
    return (rev_wet * rev_gain + dly_wet * dly_gain).astype(np.float32)


def _master_2mix(
    stereo_mix: np.ndarray,
    sr: int,
    target_lufs: float = -8.2,
) -> np.ndarray:
    """Apply cohesive 2-bus glue & true-peak limiting (`Wave Breaker` / Pedalboard) to target LUFS."""
    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(stereo_mix.T.astype(np.float64))
    gain_needed_db = float(np.clip(target_lufs - current_lufs, -6.0, 9.0)) if np.isfinite(current_lufs) else 0.0

    pmap = get_plugin_map()
    bus_scaled = (stereo_mix * (10.0 ** (gain_needed_db / 20.0))).astype(np.float32)

    if pmap["wave_breaker"].exists:
        try:
            wb = load_plugin(str(pmap["wave_breaker"].loadable_path))
            if "threshold_db" in wb.parameters:
                wb.threshold_db = -1.2
            out = wb(bus_scaled, sr)
            return np.clip(out, -0.985, 0.985).astype(np.float32)
        except Exception:
            pass

    limiter = Pedalboard(
        [
            Compressor(threshold_db=-6.0, ratio=1.6, attack_ms=25.0, release_ms=140.0),
            Limiter(threshold_db=-0.6, release_ms=60.0),
        ]
    )
    return np.clip(limiter(bus_scaled, sr), -0.985, 0.985).astype(np.float32)


def run_full_automix_for_song(
    song_folder: Path,
    template_song_path: Path,
    profiles_json_path: Path,
    output_root: Path,
    style: str = "consensus",
    target_sr: int = 44100,
) -> dict:
    """Execute the complete automated vocal mixing & Studio One 5 session generation pipeline."""
    song_name = song_folder.name
    safe_slug = song_name.replace("&", "and").replace(" ", "_")
    out_dir = output_root / f"{safe_slug}_{style}"
    media_dir = out_dir / "Media"
    media_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [1/7] Loading & consolidating stems for '{song_name}' ===", flush=True)
    assets = discover_song_assets(song_folder)
    inst_stereo, _ = load_audio(assets["inst"], target_sr=target_sr, mono=False)  # type: ignore[arg-type]
    inst_mono = np.mean(inst_stereo, axis=0, dtype=np.float32)

    consolidated = consolidate_vocal_stem_folder(assets["stems"], target_sr=target_sr)  # type: ignore[arg-type]

    # Align common timeline length
    common_len = max(inst_stereo.shape[1], len(consolidated.lead_mono))
    if inst_stereo.shape[1] < common_len:
        inst_stereo = np.pad(inst_stereo, ((0, 0), (0, common_len - inst_stereo.shape[1])))
        inst_mono = np.pad(inst_mono, (0, common_len - len(inst_mono)))
    lead_raw = np.pad(consolidated.lead_mono, (0, max(0, common_len - len(consolidated.lead_mono))))[:common_len]

    # Estimate BPM from instrumental
    tempo_est, _ = librosa.beat.beat_track(y=inst_mono[: min(len(inst_mono), target_sr * 90)], sr=target_sr)
    bpm = round(float(np.atleast_1d(tempo_est)[0]), 1)
    if bpm < 75.0:
        bpm *= 2.0
    print(f"Detected Tempo: {bpm} BPM | Duration: {common_len / target_sr:.2f}s", flush=True)

    print("=== [2/7] Stage 1 Pre-processing (RX 10 cleanup + Pitch Stabilizer + Phoneme Gain Rider + Soothe2 Polish) ===", flush=True)
    prep_cfg = PreprocessConfig()
    lead_prepared, rider_curve_db = preprocess_vocal_pipeline(lead_raw, inst_mono, sr=target_sr, cfg=prep_cfg)
    # Apply RX 10 De-ess & Oeksound Soothe2 resonance taming BEFORE optimization so CMA-ES converges on exact post-polish tone!
    default_polish = VocalChainParams(
        hpf_hz=100.0, body_gain_db=0.0, mud_cut_db=0.0, presence_gain_db=0.0, air_shelf_db=0.0,
        deess_threshold_db=-22.0, mv2_low_level=0.0, mv2_high_level=0.0, comp_threshold_db=-18.0,
        comp_ratio=3.0, soothe_depth=0.32, saturation_drive=0.0, vocal_fader_db=0.0,
        reverb_send_db=-16.0, delay_send_db=-19.0, doubler_send_db=-21.0, unmask_mid_dip_db=1.5,
    )
    lead_prepared = render_real_vst3_Polish_pass(lead_prepared, sr=target_sr, params=default_polish)

    print(f"=== [3/7] Structural Segmentation & CMA-ES Optimization (Target Style: '{style}') ===", flush=True)
    target_prof = load_target_profile(profiles_json_path, style=style)
    raw_spans, anchor_idx = detect_song_sections(lead_prepared, inst_mono, sr=target_sr)
    a_start, a_end, a_lbl = raw_spans[anchor_idx]
    s0, s1 = int(a_start * target_sr), int(a_end * target_sr)

    chain_params, achieved_anchor_prof, final_loss = optimize_vocal_chain_to_target(
        lead_prepared[s0:s1],
        inst_mono[s0:s1],
        sr=target_sr,
        target_profile=target_prof,
        n_iterations=56,
    )
    print(
        f"Anchor '{a_lbl}' ({a_start:.1f}s-{a_end:.1f}s) converged (loss={final_loss}) -> "
        f"HPF={chain_params.hpf_hz}Hz, Body={chain_params.body_gain_db:+.2f}dB, "
        f"Mud={chain_params.mud_cut_db:+.2f}dB, Presence={chain_params.presence_gain_db:+.2f}dB, "
        f"Air={chain_params.air_shelf_db:+.2f}dB, Fader={chain_params.vocal_fader_db:+.2f}dB",
        flush=True,
    )

    print("=== [4/7] Propagating Anchor Mix Balance Across All Song Sections ===", flush=True)
    # Apply core vocal chain across entire song using exact anchor-normalized gain relationship
    anchor_raw_rms = float(np.sqrt(np.mean(lead_prepared[s0:s1] ** 2) + 1e-12))
    anchor_norm_db = -18.0 - (20.0 * np.log10(anchor_raw_rms))
    lead_norm_full = (lead_prepared * (10.0 ** (anchor_norm_db / 20.0))).astype(np.float32)
    params_rel = VocalChainParams(
        **{**chain_params.to_dict(), "vocal_fader_db": chain_params.vocal_fader_db - anchor_norm_db}
    )
    lead_chained_base = apply_fast_parametric_vocal_chain(lead_norm_full, sr=target_sr, params=params_rel)

    song_sections: list[SongSection] = []
    section_envelope = np.ones(common_len, dtype=np.float32)

    for idx, (st_sec, en_sec, lbl) in enumerate(raw_spans):
        i0, i1 = int(st_sec * target_sr), int(en_sec * target_sr)
        if idx == anchor_idx or "Instrumental" in lbl:
            sec_prof = compute_section_relative_profile(lead_chained_base[i0:i1], inst_mono[i0:i1], sr=target_sr)
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
            sec_prof = compute_section_relative_profile(lead_chained_base[i0:i1], inst_mono[i0:i1], sr=target_sr)
            adj = compute_anchor_propagation(achieved_anchor_prof, sec_prof)
            # Clamp section clip gain to musical [-4.0, +3.5] dB range so soft bridges retain intimacy without vanishing
            cg_db = round(float(np.clip(adj.input_gain_delta_db * 0.65, -4.0, 3.5)), 2)
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

    # Smooth section boundary transitions (120ms crossfade via fast fftconvolve)
    xf_win = np.hanning(int(0.12 * target_sr)).astype(np.float32)
    xf_win /= np.sum(xf_win)
    section_envelope = signal.fftconvolve(section_envelope, xf_win, mode="same").astype(np.float32)
    lead_final_dry = (lead_chained_base * section_envelope).astype(np.float32)

    print("=== [5/7] Stage 3 Ethereal FX Throws & Stage 4 Harmony Director Guides ===", flush=True)
    fx_events = detect_creative_fx_events(lead_final_dry, inst_mono, sr=target_sr)
    fx_throw_stereo = render_ethereal_throw_stem(lead_final_dry, sr=target_sr, events=fx_events)
    guide_up_mono, guide_low_mono, harmony_plan = generate_harmony_guides_and_plan(
        lead_prepared, inst_mono, sr=target_sr, sections=song_sections
    )

    print("=== [6/7] Rendering Supporting Layers, Spatial Bus, Unmasked Inst & Master 2mix ===", flush=True)
    spatial_wet_stereo = _render_spatial_sends(lead_final_dry, sr=target_sr, params=chain_params, bpm=bpm)
    vocal_bus_stereo = np.stack([lead_final_dry, lead_final_dry], axis=0) + spatial_wet_stereo + fx_throw_stereo

    # Mix in user's recorded double / octave / harmony stems at balanced relative levels
    harm1_mono_for_s1: np.ndarray | None = None
    harm2_mono_for_s1: np.ndarray | None = None

    lead_active_rms = float(np.sqrt(np.mean(lead_final_dry[lead_final_dry != 0] ** 2) + 1e-12))

    def _balance_aux_stereo(aux_st: np.ndarray | None, rel_db: float, hpf: float = 170.0) -> np.ndarray | None:
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
                LowpassFilter(10500.0),
                Compressor(threshold_db=-22.0, ratio=3.8, attack_ms=8.0, release_ms=110.0),
            ]
        )
        proc = b(padded, target_sr)
        cur_rms = float(np.sqrt(np.mean(proc[np.abs(proc) > 1e-4] ** 2) + 1e-12))
        target_rms = lead_active_rms * (10.0 ** (rel_db / 20.0))
        gain = min(target_rms / cur_rms, 4.0)
        return (proc * gain).astype(np.float32)

    if consolidated.double_mono is not None:
        dbl_st = np.stack([consolidated.double_mono, consolidated.double_mono], axis=0)
        dbl_bal = _balance_aux_stereo(dbl_st, rel_db=-9.5, hpf=180.0)
        if dbl_bal is not None:
            vocal_bus_stereo += dbl_bal

    for oc_st, rel_db, hpf_hz in [
        (consolidated.sub_octave_stereo, -12.5, 95.0),
        (consolidated.super_octave_stereo, -13.0, 240.0),
    ]:
        oc_bal = _balance_aux_stereo(oc_st, rel_db=rel_db, hpf=hpf_hz)
        if oc_bal is not None:
            vocal_bus_stereo += oc_bal

    if len(consolidated.harmonies_stereo) >= 1:
        h1_bal = _balance_aux_stereo(consolidated.harmonies_stereo[0], rel_db=-8.5, hpf=180.0)
        if h1_bal is not None:
            vocal_bus_stereo += h1_bal
            harm1_mono_for_s1 = np.mean(consolidated.harmonies_stereo[0], axis=0)[:common_len]
    if len(consolidated.harmonies_stereo) >= 2:
        h2_bal = _balance_aux_stereo(consolidated.harmonies_stereo[1], rel_db=-9.0, hpf=180.0)
        if h2_bal is not None:
            vocal_bus_stereo += h2_bal
            harm2_mono_for_s1 = np.mean(consolidated.harmonies_stereo[1], axis=0)[:common_len]

    inst_unmasked = apply_dynamic_instrumental_unmasking(
        inst_stereo, lead_final_dry, sr=target_sr, max_dip_db=chain_params.unmask_mid_dip_db
    )
    pre_master_2mix = inst_unmasked + vocal_bus_stereo
    final_master_2mix = _master_2mix(pre_master_2mix, sr=target_sr, target_lufs=-8.2)

    # Write Media WAVs for Studio One 5 session & user listening verification
    inst_wav = media_dir / "01_offvocal_unmasked.wav"
    main_prep_wav = media_dir / "02_mainvocal_prepared.wav"
    harm1_wav = media_dir / "03_harmvocal_1.wav"
    harm2_wav = media_dir / "04_harmvocal_2.wav"
    throws_wav = media_dir / "05_fx_ethereal_throws.wav"
    guide_up_wav = media_dir / "06_director_guide_upper3rd.wav"
    guide_low_wav = media_dir / "07_director_guide_lower3rd.wav"

    write_pcm24_wav(inst_wav, inst_unmasked, target_sr)
    write_pcm24_wav(main_prep_wav, lead_prepared, target_sr)
    if harm1_mono_for_s1 is not None:
        write_pcm24_wav(harm1_wav, harm1_mono_for_s1, target_sr)
    if harm2_mono_for_s1 is not None:
        write_pcm24_wav(harm2_wav, harm2_mono_for_s1, target_sr)
    write_pcm24_wav(throws_wav, fx_throw_stereo, target_sr)
    write_pcm24_wav(guide_up_wav, guide_up_mono, target_sr)
    write_pcm24_wav(guide_low_wav, guide_low_mono, target_sr)

    master_wav = out_dir / f"{safe_slug}_AutoMIX_Master.wav"
    vocal_only_wav = out_dir / f"{safe_slug}_AutoMIX_VocalBus.wav"
    write_pcm24_wav(master_wav, final_master_2mix, target_sr)
    write_pcm24_wav(vocal_only_wav, vocal_bus_stereo, target_sr)

    print("=== [7/7] Exporting Optimized VST3 Presets & Building Studio One 5 .song Session ===", flush=True)
    patched_presets = export_template_vst3_states(chain_params, bpm=bpm)
    extra_tracks = [
        TrackMediaSpec(name="offvocal", wav_path=inst_wav, stereo=True, color="FF2D7DD2", gain_db=0.0, mute=False),
        TrackMediaSpec(
            name="fx_ethereal_throws", wav_path=throws_wav, stereo=True, color="FF9B51E0", gain_db=0.0, mute=False
        ),
        TrackMediaSpec(
            name="guide_upper3rd (Muted)",
            wav_path=guide_up_wav,
            stereo=False,
            color="FF27AE60",
            gain_db=-4.0,
            mute=True,
        ),
        TrackMediaSpec(
            name="guide_lower3rd (Muted)",
            wav_path=guide_low_wav,
            stereo=False,
            color="FF16A085",
            gain_db=-4.5,
            mute=True,
        ),
    ]

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
        chain_params=chain_params,
    )

    # Objective evaluation: compare Full AutoMIX vs Pro Target (and vs User's Self-Mix if available!)
    final_full_prof = compute_section_relative_profile(
        np.mean(vocal_bus_stereo, axis=0), np.mean(inst_unmasked, axis=0), sr=target_sr
    )
    final_anchor_bus_prof = compute_section_relative_profile(
        np.mean(vocal_bus_stereo[:, s0:s1], axis=0),
        np.mean(inst_unmasked[:, s0:s1], axis=0),
        sr=target_sr,
    )

    self_mix_prof_dict = None
    if assets["self_mix"] is not None:
        try:
            from mix_automation.analysis.reference_extractor import GPUSourceSeparator

            sep = GPUSourceSeparator(cache_dir=output_root.parent / "processed" / "self_mix_sep")
            sv, si, _, _ = sep.separate_to_vocal_and_inst(
                assets["self_mix"], slug=f"selfmix_{safe_slug}", target_sr=target_sr  # type: ignore[arg-type]
            )
            sa0 = min(s0, max(0, len(sv) - target_sr * 10))
            sa1 = min(s1, len(sv))
            self_mix_prof_dict = compute_section_relative_profile(sv[sa0:sa1], si[sa0:sa1], sr=target_sr).to_dict()
        except Exception:
            self_mix_prof_dict = None

    report = {
        "song_name": song_name,
        "style_target": style,
        "bpm": bpm,
        "sample_rate": target_sr,
        "duration_sec": round(common_len / target_sr, 2),
        "studio_one_song": str(song_file),
        "master_wav": str(master_wav),
        "vocal_bus_wav": str(vocal_only_wav),
        "optimized_chain_params": chain_params.to_dict(),
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
            "automix_dry_core": achieved_anchor_prof.to_dict(),
            "automix_full_vocal_bus": final_anchor_bus_prof.to_dict(),
            "user_self_mix": self_mix_prof_dict,
        },
        "automix_full_song_profile": final_full_prof.to_dict(),
        "stage3_creative_fx_events": [ev.to_dict() for ev in fx_events],
        "stage4_director_harmony_plan": [hs.to_dict() for hs in harmony_plan],
    }

    report_json_path = out_dir / "mix_report.json"
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_director_markdown_report(out_dir / "MIX_DIRECTOR_REPORT.md", report)

    return report


def _write_director_markdown_report(md_path: Path, report: dict) -> None:
    """Write a human-readable Japanese Director & Mix Settings Report."""
    p = report["optimized_chain_params"]
    comp = report["comparison_anchor_chorus"]
    tgt = comp["pro_target"]
    ach = comp["automix_dry_core"]
    slf = comp.get("user_self_mix")

    lines = [
        f"# 自動MIX＆ディレクション診断レポート — {report['song_name']}",
        "",
        f"- **ターゲットMIXスタイル**: `{report['style_target']}`",
        f"- **検出テンポ**: `{report['bpm']} BPM` （尺: `{report['duration_sec']}秒`）",
        f"- **生成済みStudio One 5 プロジェクト**: `{report['studio_one_song']}`",
        f"- **完成マスタリング音源**: `{report['master_wav']}`",
        "",
        "## 1. Studio One 5 設定パラメータ（代表サビ最適化値）",
        "",
        "| プラグイン / 処理 | パラメータ | 算出値 | 役割 |",
        "|---|---|---|---|",
        f"| **MV2 Mono** | Low Level / High Level | `{p['mv2_low_level']:+.1f} dB` / `{p['mv2_high_level']:+.1f} dB` | 小さい声の押し上げと濃密な張り付き |",
        f"| **Persistent Q** | Band 1 Highpass (18dB/oct) | `{p['hpf_hz']} Hz` | ローカット・オケ低域との分離 |",
        f"| **Persistent Q** | Band 2 Body (240 Hz, Q=0.9) | `{p['body_gain_db']:+.2f} dB` | オケに負けない声の太さ・土台 |",
        f"| **Persistent Q** | Band 3 Mud (560 Hz, Q=1.4) | `{p['mud_cut_db']:+.2f} dB` | こもり・箱鳴りの整理 |",
        f"| **Persistent Q** | Band 4 Presence (2850 Hz, Q=0.85) | `{p['presence_gain_db']:+.2f} dB` | 前後の距離感・最前面への張り付き |",
        f"| **Persistent Q** | Band 5 Air Shelf (10.5 kHz) | `{p['air_shelf_db']:+.2f} dB` | 上級者特有のシルキーな抜け感 |",
        f"| **C1 comp Mono** | Threshold / Ratio | `{p['comp_threshold_db']:+.1f} dB` / `{p['comp_ratio']}:1` | ピーク制御と位置の安定 |",
        f"| **RX 10 De-ess / Soothe2** | De-ess Thresh / Soothe Depth | `{p['deess_threshold_db']:+.1f} dB` / `{p['soothe_depth']:.2f}` | 歯擦音と2〜5kHzの刺さり共振の除去 |",
        f"| **FX Sends** | Reverb(R2) / H-Delay / Doubler2 | `{p['reverb_send_db']:+.1f}` / `{p['delay_send_db']:+.1f}` / `{p['doubler_send_db']:+.1f} dB` | 空間の馴染みと立体感 |",
        "",
        "## 2. 対オケ相対バランス比較（基準サビ区間）",
        "",
        "| 指標 | 上級者ターゲット | 今回のAutoMIX | ご自身の過去MIX |",
        "|---|---|---|---|",
        f"| 対オケ総合ラウドネス比 | `{tgt['broadband_vi_ratio_db']:+.2f} dB` | `{ach['broadband_vi_ratio_db']:+.2f} dB` | `{slf['broadband_vi_ratio_db']:+.2f} dB` |" if slf else f"| 対オケ総合ラウドネス比 | `{tgt['broadband_vi_ratio_db']:+.2f} dB` | `{ach['broadband_vi_ratio_db']:+.2f} dB` | - |",
        f"| 芯の太さ (160-420Hz) | `{tgt['band_margins_db']['body_warmth']:+.2f} dB` | `{ach['band_margins_db']['body_warmth']:+.2f} dB` | `{slf['band_margins_db']['body_warmth']:+.2f} dB` |" if slf else f"| 芯の太さ (160-420Hz) | `{tgt['band_margins_db']['body_warmth']:+.2f} dB` | `{ach['band_margins_db']['body_warmth']:+.2f} dB` | - |",
        f"| 張り付き存在感 (1.5-4.2kHz) | `{tgt['band_margins_db']['core_presence']:+.2f} dB` | `{ach['band_margins_db']['core_presence']:+.2f} dB` | `{slf['band_margins_db']['core_presence']:+.2f} dB` |" if slf else f"| 張り付き存在感 (1.5-4.2kHz) | `{tgt['band_margins_db']['core_presence']:+.2f} dB` | `{ach['band_margins_db']['core_presence']:+.2f} dB` | - |",
        f"| 空気感・抜け (8.5-16kHz) | `{tgt['band_margins_db']['air_sheen']:+.2f} dB` | `{ach['band_margins_db']['air_sheen']:+.2f} dB` | `{slf['band_margins_db']['air_sheen']:+.2f} dB` |" if slf else f"| 空気感・抜け (8.5-16kHz) | `{tgt['band_margins_db']['air_sheen']:+.2f} dB` | `{ach['band_margins_db']['air_sheen']:+.2f} dB` | - |",
        f"| 位置のブレ幅 (小さいほど安定) | `{tgt['position_stability_std_db']:.2f} dB` | `{ach['position_stability_std_db']:.2f} dB` | `{slf['position_stability_std_db']:.2f} dB` |" if slf else f"| 位置のブレ幅 (小さいほど安定) | `{tgt['position_stability_std_db']:.2f} dB` | `{ach['position_stability_std_db']:.2f} dB` | - |",
        "",
        "## 3. セクション別アンカー伝播（パート間音圧追従クリップゲイン）",
        "",
        "| セクション | 時間区間 | オケRMS | 自動適用クリップゲイン |",
        "|---|---|---|---|",
    ]
    for s in report["sections"]:
        mark = " **(基準アンカー)**" if s["is_anchor"] else ""
        lines.append(
            f"| `{s['label']}`{mark} | `{s['start_sec']:.1f}s - {s['end_sec']:.1f}s` | `{s['inst_rms_db']:.1f} dB` | `{s['clip_gain_db']:+.2f} dB` |"
        )

    lines.extend(["", "## 4. 目標③：検出された特殊空間演出（儚いリバーブ投げ等）", ""])
    if report["stage3_creative_fx_events"]:
        for ev in report["stage3_creative_fx_events"][:12]:
            lines.append(f"- **`{ev['start_sec']:.2f}s - {ev['end_sec']:.2f}s`** (`{ev['fx_type']}`): {ev['reason']}")
    else:
        lines.append("- 特筆すべき局所特殊エフェクト箇所なし（標準空間センドで処理）")

    lines.extend(["", "## 5. 目標④：ディレクターによるコーラス・ハモリ構成提案", ""])
    for hs in report["stage4_director_harmony_plan"]:
        layers_str = " / ".join(hs["recommended_layers"])
        lines.append(
            f"- **`{hs['section_label']}` (`{hs['start_sec']:.1f}s - {hs['end_sec']:.1f}s` — {hs['arrangement_role']})**: 推奨レイヤー `[{layers_str}]` — {hs['director_note_ja']}"
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
    root = ET.fromstring(xml_text)

    # Find mainvocal MediaTrack events and read user-edited `volumeCurve` level on anchor_event_index
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

    # Shift all non-instrumental events by the exact relative offset the user dialed in!
    out_path = song_path.with_name(song_path.stem + "_propagated.song")
    updated_xml = xml_text
    if orig_sections:
        for s in orig_sections:
            ev_name = f"main_{s['label']}"
            old_lin = round(float(10.0 ** (s["clip_gain_db"] / 20.0)), 6)
            new_db = s["clip_gain_db"] + (0.0 if "Instrumental" in s["label"] else user_delta_db)
            new_lin = round(float(10.0 ** (new_db / 20.0)), 6)
            # Replace volumeCurve level inside that specific named AudioEvent block
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
