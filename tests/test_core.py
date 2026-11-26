"""v2 Unit tests for acoustic profiling, Stage 3 evidence gating, chord-safe harmonies & Studio One .song XML."""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mido
import numpy as np
from pedalboard import Delay, Pedalboard
from scipy import signal

from mix_automation.analysis.preference_learner import (
    apply_user_preference_offsets,
    propagate_anchor_advanced,
    update_user_preference_offsets,
)
from mix_automation.analysis.reference_extractor import (
    synthesize_acoustic_knn_target,
    validate_reference_song_candidate,
)
from mix_automation.analysis.relative_profile import (
    classify_backing_archetype,
    compute_anchor_propagation,
    compute_section_relative_profile,
)
from mix_automation.creative.fx_and_harmony import (
    _select_and_stitch_chord_safe_harmony,
    apply_verified_inline_fx_chain,
    detect_creative_fx_events,
    generate_counter_melody_and_export_midi,
    render_spot_delay_throw_stem,
)
from mix_automation.dsp.lyrics_aligner import (
    align_lyrics_and_phonemes,
    find_lyric_snippet_at_span,
)
from mix_automation.dsp.pitch_timing import (
    ReviewMarker,
    evaluate_raw_vocal_correction_benchmark,
)
from mix_automation.dsp.preprocess import (
    PreprocessConfig,
    preprocess_vocal_pipeline,
)
from mix_automation.dsp.vst3_optimizer import (
    SongSection,
    VocalChainParams,
    apply_bark_consonant_transient_tamer,
    apply_dynamic_instrumental_unmasking,
    apply_multisection_adaptive_sculpting,
    compute_continuous_distance_envelopes,
    detect_song_sections,
    detect_vocal_surgical_notches,
    estimate_continuous_tempo_map,
    process_choir_bus,
    process_dedicated_harmony_bus,
    process_main_harmony_bus,
    solve_differentiable_intrinsic_eq_gpu,
    solve_precision_10band_eq_gpu,
)
from mix_automation.io.audio_loader import consolidate_vocal_stem_folder
from mix_automation.io.song_builder import (
    DEFAULT_S1_PLUGIN_SPECS,
    TrackMediaSpec,
    build_studio_one_song,
    create_vstpreset_bytes,
    extract_steinberg_comp_and_cont,
    patch_vstpreset_component_state,
    write_pcm24_wav,
)
from mix_automation.io.studio_one import write_vstpreset
from mix_automation.pipeline import discover_song_assets


def test_relative_profile_and_anchor_propagation() -> None:
    sr = 44100
    t = np.linspace(0.0, 4.0, sr * 4, endpoint=False)

    # Modulated vocal envelope (4 Hz syllabic rhythm) + stereo width
    syllabic_env = (0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)
    v_dry = syllabic_env * (0.25 * np.sin(2 * np.pi * 260 * t) + 0.15 * np.sin(2 * np.pi * 2400 * t))
    vocal_st = np.stack([v_dry * 1.02, v_dry * 0.98], axis=0).astype(np.float32)

    inst_m = (0.30 * np.sin(2 * np.pi * 120 * t) + 0.25 * np.sin(2 * np.pi * 2200 * t)).astype(np.float32)
    inst_st = np.stack([inst_m, inst_m], axis=0)

    anchor = compute_section_relative_profile(vocal_st, inst_st, sr=sr)
    assert anchor.syllabic_clarity_index > 0.10
    assert anchor.backing_archetype in {"dense_rock_edm", "groovy_mid_pop", "intimate_ballad_sparse"}

    inst_verse = inst_st * 0.25
    raw_verse = compute_section_relative_profile(vocal_st, inst_verse, sr=sr)
    adjustment = compute_anchor_propagation(anchor, raw_verse)
    assert -13.0 <= adjustment.input_gain_delta_db <= -11.0


def test_stage3_fx_requires_reference_vocal_evidence() -> None:
    sr = 22050
    dur = 12.0
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)

    # Create 5 phrases of 0.8s followed by 1.2s silent rests
    lead = np.zeros(n, dtype=np.float32)
    ref_dry = np.zeros(n, dtype=np.float32)
    for k in range(5):
        s0 = int((0.5 + k * 2.1) * sr)
        s1 = s0 + int(0.8 * sr)
        tone = (0.25 * np.sin(2 * np.pi * 330 * t[s0:s1])).astype(np.float32)
        lead[s0:s1] = tone
        ref_dry[s0:s1] = tone * 0.95 + 1e-4 * np.random.default_rng(k).standard_normal(s1 - s0).astype(np.float32)

    inst = (0.1 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)

    # 1. Without reference vocal -> NEVER trigger blind reverb throws!
    assert detect_creative_fx_events(lead, inst, sr=sr, ref_vocal_mono=None) == []

    # 2. Reference vocal has dry rests everywhere -> 0 throws!
    assert detect_creative_fx_events(lead, inst, sr=sr, ref_vocal_mono=ref_dry) == []

    # 3. Inject genuine decaying reverb tail ONLY after phrase #2 (index k=2, end at 0.5 + 4.2 + 0.8 = 5.5s)
    ref_with_throw = ref_dry.copy()
    tail_s0 = int(5.5 * sr)
    tail_len = int(0.65 * sr)
    decay = np.exp(-np.linspace(0.0, 3.2, tail_len)).astype(np.float32)
    ref_with_throw[tail_s0 : tail_s0 + tail_len] += (
        0.09 * decay * np.sin(2 * np.pi * 330 * t[tail_s0 : tail_s0 + tail_len]).astype(np.float32)
    )

    events = detect_creative_fx_events(lead, inst, sr=sr, ref_vocal_mono=ref_with_throw)
    throws = [e for e in events if e.fx_type == "ethereal_reverb_throw"]
    assert len(throws) == 1
    assert 5.0 <= throws[0].start_sec <= 5.6


def test_chord_safe_harmony_rescues_borrowed_chord_clash() -> None:
    sr = 22050
    hop = 512
    n_frames = 40
    n_samples = n_frames * hop

    # Global scale = C Major (C, D, E, F, G, A, B -> pitch classes 0,2,4,5,7,9,11)
    c_major = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    local_scale = np.tile(c_major[:, None], (1, n_frames))

    # Backing chord is secondary dominant E7 (E=4, G#=8, B=11) -> G# clashes by 1 semitone with G(7)!
    chord_chroma = np.zeros((12, n_frames), dtype=np.float32)
    chord_chroma[4, :] = 1.0   # E
    chord_chroma[8, :] = 0.95  # G# (borrowed major 3rd of E7!)
    chord_chroma[11, :] = 0.85 # B

    # Singer sings E4 (MIDI 64, pitch class 4 = E).
    # Naive C-major +3rd on E is +3 semitones -> G (pc=7), which clashes harshly against G# (pc=8)!
    # Chord-safe resolver must rescue this note to +4 semitones -> G# (pc=8)!
    lead_f0 = np.full(n_frames, 329.63, dtype=np.float32)  # E4
    voiced_mask = np.ones(n_frames, dtype=bool)

    candidates = {
        3: np.full(n_samples, 0.3, dtype=np.float32),
        4: np.full(n_samples, 0.4, dtype=np.float32),
        5: np.full(n_samples, 0.5, dtype=np.float32),
    }

    stitched, n_notes, rescued = _select_and_stitch_chord_safe_harmony(
        candidates,
        allowed_intervals=(3, 4, 5),
        lead_f0=lead_f0,
        voiced_mask=voiced_mask,
        chord_chroma=chord_chroma,
        local_scale=local_scale,
        sr=sr,
        hop=hop,
    )
    assert n_notes == 1
    assert rescued == 1
    # Output amplitude should match candidate[4] (= 0.4), NOT candidate[3] (= 0.3)!
    assert np.allclose(stitched[n_samples // 4 : 3 * n_samples // 4], 0.4, atol=1e-2)


def test_write_and_patch_vstpreset_roundtrip(tmp_path: Path) -> None:
    dummy_cid = "0123456789ABCDEF0123456789ABCDEF"
    orig_state = b"<VST3State value='1'/>\x00"
    preset_file = write_vstpreset(tmp_path / "test.vstpreset", dummy_cid, orig_state)
    orig_bytes = preset_file.read_bytes()

    new_state = b"<VST3State value='999' updated='true'/>\x00"
    patched = patch_vstpreset_component_state(orig_bytes, new_state)

    magic, version, cid_bytes, list_offset = struct.unpack("<4si32sq", patched[:48])
    assert magic == b"VST3"
    assert version == 1
    assert cid_bytes.decode("ascii") == dummy_cid
    assert patched[48 : 48 + len(new_state)] == new_state
    assert patched[list_offset : list_offset + 4] == b"List"


def test_studio_one_song_builder_beat_offsets_and_markers(tmp_path: Path) -> None:
    template_path = Path(r"C:\MIX_automation\data\templates\test20261124\test20261124.song")
    if not template_path.exists():
        return

    sr = 44100
    bpm = 146.0
    sec_per_beat = 60.0 / bpm
    wav_path = tmp_path / "main.wav"
    write_pcm24_wav(wav_path, np.zeros(sr * 24, dtype=np.float32), sr)

    sections = [
        SongSection(0, "01_Verse", 0.0, 10.0, False, -14.0, -1.2, None),
        SongSection(1, "02_Chorus", 10.0, 22.0, True, -10.0, 0.0, None),
    ]
    markers = [ReviewMarker(time_sec=12.5, name="ピッチ確認 (+48ct)", category="pitch")]

    out_song = build_studio_one_song(
        template_song_path=template_path,
        output_song_dir=tmp_path,
        song_name="UnitCheck",
        bpm=bpm,
        sr=sr,
        main_vocal_wav=wav_path,
        sections=sections,
        extra_tracks=[],
        review_markers=markers,
    )

    with zipfile.ZipFile(out_song, "r") as zf:
        song_xml = zf.read("Song/song.xml").decode("utf-8")

    parseable_xml = song_xml.lstrip("\ufeff").replace(
        "<Song ", '<Song xmlns:x="http://www.presonus.com/xml" ', 1
    )
    root = ET.fromstring(parseable_xml)
    # Verify second AudioEvent has both start AND offset in quarter-note beats (10.0 / sec_per_beat = 24.33333333)
    expected_beats = 10.0 / sec_per_beat
    audio_events = [
        ev for m in root.iter("MediaTrack") if m.attrib.get("name") == "mainvocal" for ev in m.iter("AudioEvent")
    ]
    assert len(audio_events) == 2
    ev2 = audio_events[1]
    assert abs(float(ev2.attrib["start"]) - expected_beats) < 1e-4
    assert abs(float(ev2.attrib["offset"]) - expected_beats) < 1e-4

    # Verify MarkerTrack is visible (hidden="0") and contains our ReviewMarker
    mtrack = root.find(".//MarkerTrack")
    assert mtrack is not None
    marker_names = [me.attrib.get("name", "") for me in mtrack.iter("MarkerEvent")]
    assert any("ピッチ確認" in n for n in marker_names)


def test_detect_song_sections_identifies_loudest_chorus() -> None:
    sr = 22050
    t_seg = np.linspace(0.0, 18.0, sr * 18, endpoint=False)
    verse_i = 0.08 * np.sin(2 * np.pi * 220 * t_seg)
    chorus_i = 0.45 * np.sin(2 * np.pi * 220 * t_seg) + 0.25 * np.sin(2 * np.pi * 1760 * t_seg)
    inst = np.concatenate([verse_i, chorus_i]).astype(np.float32)
    vocal = (0.20 * np.sin(2 * np.pi * 330 * np.linspace(0.0, 36.0, sr * 36, endpoint=False))).astype(np.float32)

    sections, anchor_idx = detect_song_sections(vocal, inst, sr=sr, min_section_sec=10.0)
    assert len(sections) >= 2
    _, a_end, _ = sections[anchor_idx]
    assert a_end > 25.0


def test_acoustic_dedup_and_plus3_harmony_classification(tmp_path: Path) -> None:
    sr = 22050
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    main_sig = (0.25 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    harm_sig = (0.008 * np.sin(2 * np.pi * 415 * t)).astype(np.float32)  # Quiet -45 dBFS harmony!

    p_main = tmp_path / "song - main1.wav"
    p_dup_st = tmp_path / "song - choir1.wav"
    p_mono_ch = tmp_path / "songmono - choir1.wav"
    p_plus3 = tmp_path / "songmono - +3 sabi.wav"
    p_minus3 = tmp_path / "songmono - -3.wav"

    write_pcm24_wav(p_main, main_sig, sr)
    write_pcm24_wav(p_dup_st, np.stack([harm_sig, harm_sig], axis=0), sr)  # Dual-mono duplicate
    write_pcm24_wav(p_mono_ch, harm_sig, sr)  # Canonical 1ch mono re-export
    write_pcm24_wav(p_plus3, (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    write_pcm24_wav(p_minus3, (0.01 * np.sin(2 * np.pi * 277 * t)).astype(np.float32), sr)

    cons = consolidate_vocal_stem_folder([p_main, p_dup_st, p_mono_ch, p_plus3, p_minus3], target_sr=sr)
    assert "song - choir1.wav" in cons.deduplicated_files
    assert cons.has_user_harmonies
    assert cons.user_harmony_track_count == 3  # choir1(mono) + +3 sabi + -3 (duplicate dropped!)
    assert cons.harmonies_stereo_bus is not None
    # Verify quiet -45 dBFS harmonies were leveled up into healthy active studio range!
    assert float(np.sqrt(np.mean(cons.harmonies_stereo_bus**2))) > 0.03


def test_gpu_differentiable_intrinsic_eq_warmstart() -> None:
    sr = 44100
    rng = np.random.default_rng(123)
    # Dark muffled vocal (excess 560Hz mud, deficient 2.9kHz presence)
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    dark_v = (
        0.25 * np.sin(2 * np.pi * 560 * t)
        + 0.08 * np.sin(2 * np.pi * 240 * t)
        + 0.02 * np.sin(2 * np.pi * 2900 * t)
        + 0.01 * rng.standard_normal(len(t))
    ).astype(np.float32)
    inst = (0.15 * rng.standard_normal(len(t))).astype(np.float32)

    prof = compute_section_relative_profile(dark_v, inst, sr=sr)
    # Construct pro-like intrinsic target with stronger presence & cleaner mud
    tgt = type(prof)(
        **{
            **prof.to_dict(),
            "vocal_intrinsic_bands_db": {
                "sub_low": -35.0,
                "body_warmth": -6.0,
                "boxiness_mud": -4.2,
                "core_mid": -6.1,
                "core_presence": -9.3,
                "sibilance": -18.0,
                "air_sheen": -19.5,
            },
        }
    )
    eq = solve_differentiable_intrinsic_eq_gpu(dark_v, sr=sr, target_profile=tgt, inst_high_rolloff_db=-9.0)
    # Should cut excessive 560Hz mud and boost deficient 2.9kHz presence!
    assert eq["mud_cut_db"] < -1.0
    assert eq["presence_gain_db"] > 1.0


def test_bark_adaptive_spectral_unmasking_preserves_side_and_carves_mid() -> None:
    sr = 44100
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    # Backing track has strong 2.2 kHz Mid signal colliding with vocal + distinct 1.1 kHz Side signal
    mid_sig = (0.30 * np.sin(2 * np.pi * 2200 * t) + 0.25 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)
    side_sig = (0.18 * np.sin(2 * np.pi * 1100 * t)).astype(np.float32)
    inst_st = np.stack([mid_sig + side_sig, mid_sig - side_sig], axis=0)
    # Active vocal sings strongly at 2.2 kHz
    voc = (0.28 * np.sin(2 * np.pi * 2200 * t)).astype(np.float32)

    out_st = apply_dynamic_instrumental_unmasking(inst_st, voc, sr=sr, max_dip_db=3.0)
    out_mid = 0.5 * (out_st[0] + out_st[1])
    out_side = 0.5 * (out_st[0] - out_st[1])

    # Side channel must remain 100% untouched!
    assert np.allclose(out_side, side_sig, atol=1e-5)
    # Mid 2.2 kHz collision band must be measurably carved while 90 Hz sub-bass stays untouched!
    f, p_in = np.fft.rfftfreq(len(mid_sig), 1 / sr), np.abs(np.fft.rfft(mid_sig)) ** 2
    p_out = np.abs(np.fft.rfft(out_mid)) ** 2
    dip_2k2_db = 10.0 * np.log10((np.sum(p_out[(f > 2000) & (f < 2400)]) + 1e-12) / (np.sum(p_in[(f > 2000) & (f < 2400)]) + 1e-12))
    dip_sub_db = 10.0 * np.log10((np.sum(p_out[(f > 70) & (f < 110)]) + 1e-12) / (np.sum(p_in[(f > 70) & (f < 110)]) + 1e-12))
    assert dip_2k2_db < -1.0
    assert abs(dip_sub_db) < 0.15


def test_acoustic_knn_and_full_song_archetype_classification() -> None:
    import json

    profiles_path = Path(r"C:\MIX_automation\profiles\pro_target_profiles.json")
    if not profiles_path.exists():
        return
    data = json.loads(profiles_path.read_text(encoding="utf-8"))
    songs = data.get("songs", [])
    if len(songs) < 10:
        return

    # Sparse ballad query vs dense rock query
    ballad_arch = classify_backing_archetype(
        inst_crest_db=11.0,
        inst_band_levels_db={"core_presence": -30.0, "body_warmth": -24.0},
        full_inst_mid_density_index=0.14,
        backing_dynamic_range_db=7.5,
        full_inst_crest_db=11.4,
        chorus_inst_mid_density_index=0.25,
    )
    rock_arch = classify_backing_archetype(
        inst_crest_db=9.0,
        inst_band_levels_db={"core_presence": -24.0, "body_warmth": -24.0},
        full_inst_mid_density_index=0.58,
        backing_dynamic_range_db=2.2,
        full_inst_crest_db=9.5,
        chorus_inst_mid_density_index=0.68,
    )
    assert ballad_arch == "intimate_ballad_sparse"
    assert rock_arch == "dense_rock_edm"

    # Run k-NN synthesis on a real sparse ballad profile from the dataset
    sample_prof = compute_section_relative_profile(
        np.random.default_rng(1).standard_normal(44100).astype(np.float32) * 0.1,
        np.random.default_rng(2).standard_normal((2, 44100)).astype(np.float32) * 0.1,
        sr=44100,
    )
    sparse_desc = {"full_inst_mid_density_index": 0.15, "backing_dynamic_range_db": 7.8}
    knn_prof, neighbors, v_shift = synthesize_acoustic_knn_target(
        songs, sample_prof, sparse_desc, query_median_f0_hz=255.0, k=7
    )
    assert len(neighbors) >= 5
    assert 0.0 <= v_shift <= 1.35
    assert "knn_" in knn_prof.backing_archetype


def test_studio_one_v4_true_1to1_vst3_injection(tmp_path: Path) -> None:
    template_path = Path(r"C:\MIX_automation\data\templates\test20261124\test20261124.song")
    if not template_path.exists():
        return

    sr = 44100
    wav_path = tmp_path / "main.wav"
    write_pcm24_wav(wav_path, np.zeros(sr * 10, dtype=np.float32), sr)

    dummy_presets = {
        "Presets/Channels/mainvocal/1 - Persistent Q.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["persistent_q"].class_id, b"PQ_STATE"
        ),
        "Presets/Channels/mainvocal/2 - VCA Compressor.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["vca_compressor"].class_id, b"VCA_STATE"
        ),
        "Presets/Channels/mainvocal/3 - UADx LA-2A Tube Compressor.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["uad_la2a"].class_id, b"LA2A_STATE"
        ),
        "Presets/Channels/mainvocal/4 - TDR VOS SlickEQ.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["slick_eq"].class_id, b"SLICK_STATE"
        ),
        "Presets/Channels/harmvocal/1 - Persistent Q.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["persistent_q"].class_id, b"HPQ_STATE"
        ),
        "Presets/Channels/harmvocal/2 - VCA Compressor.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["vca_compressor"].class_id, b"HVCA_STATE"
        ),
        "Presets/Channels/harmvocal2/1 - Persistent Q.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["persistent_q"].class_id, b"HPQ_STATE"
        ),
        "Presets/Channels/harmvocal2/2 - VCA Compressor.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["vca_compressor"].class_id, b"HVCA_STATE"
        ),
        "Presets/Channels/Saturation Parallel/1 - Saturation Knob.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["saturation_knob"].class_id, b"SAT_STATE"
        ),
        "Presets/Channels/Phoenix EarlyRoom/1 - PhoenixVerb.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["phoenix_verb"].class_id, b"PHX_STATE"
        ),
        "Presets/Channels/メイン/1 - VCA Compressor.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["vca_compressor"].class_id, b"MVCA_STATE"
        ),
        "Presets/Channels/メイン/2 - Wave Breaker.vstpreset": create_vstpreset_bytes(
            DEFAULT_S1_PLUGIN_SPECS["wave_breaker"].class_id, b"WB_STATE"
        ),
    }

    params = VocalChainParams(
        archetype_chain="intimate_ballad_sparse",
        selected_plugins=["Persistent Q", "VCA Compressor", "UADx LA-2A", "TDR SlickEQ"],
        hpf_hz=95.0,
        body_gain_db=0.5,
        mud_cut_db=-2.0,
        core_mid_gain_db=-0.5,
        presence_gain_db=1.5,
        air_shelf_db=2.0,
        vca_threshold_db=-24.0,
        vca_measured_gr_db=2.5,
        comp2_param=38.0,
        comp2_measured_gr_db=1.5,
        parallel_sat_amount=1.4,
        parallel_sat_wet_pct=10.0,
        vocal_fader_db=0.0,
        early_room_send_db=-18.5,
        slap_delay_send_db=-19.5,
        tempo_delay_send_db=-21.0,
        plate_reverb_send_db=-20.0,
        doubler_send_db=-22.0,
        unmask_mid_dip_db=1.2,
    )

    out_song = build_studio_one_song(
        template_song_path=template_path,
        output_song_dir=tmp_path,
        song_name="True1to1Check",
        bpm=96.0,
        sr=sr,
        main_vocal_wav=wav_path,
        sections=[SongSection(0, "01_Chorus", 0.0, 10.0, True, -15.0, 0.0, None)],
        extra_tracks=[],
        patched_presets=dummy_presets,
        chain_params=params,
    )

    with zipfile.ZipFile(out_song, "r") as zf:
        names = set(zf.namelist())
        mixer_xml = zf.read("Devices/audiomixer.xml").decode("utf-8")
        console_xml = zf.read("Devices/mixerconsole.xml").decode("utf-8")

    # Verify XML parses cleanly with zero syntax corruption!
    ET.fromstring(mixer_xml.lstrip("\ufeff").replace("<AudioMixer>", '<AudioMixer xmlns:x="http://www.presonus.com/xml">', 1))
    ET.fromstring(console_xml.lstrip("\ufeff").replace("<MixerConsole>", '<MixerConsole xmlns:x="http://www.presonus.com/xml">', 1))

    assert "VCA Compressor" in mixer_xml
    assert "UADx LA-2A Tube Compressor" in mixer_xml
    assert "TDR VOS SlickEQ" in mixer_xml
    assert "Saturation Parallel" in mixer_xml
    assert "Phoenix EarlyRoom" in mixer_xml
    assert "Wave Breaker" in mixer_xml
    assert "L1 limiter Mono" not in mixer_xml  # Legacy template L1 cleanly stripped off harmonies!
    for p_rel in dummy_presets:
        assert p_rel in names


def test_juce_to_steinberg_vstpreset_unwrapping() -> None:
    pq_path = Path(r"C:\Program Files\Common Files\VST3\Persistent Q.vst3")
    if not pq_path.exists():
        return
    from pedalboard import load_plugin

    p = load_plugin(str(pq_path))
    p.band_2_gain = -4.25
    raw_juce = p.raw_state
    assert b"<VST3PluginState>" in raw_juce

    comp, _ = extract_steinberg_comp_and_cont(raw_juce)
    assert b"<VST3PluginState>" not in comp
    assert b"PersistantQState" in comp

    preset_bytes = create_vstpreset_bytes(DEFAULT_S1_PLUGIN_SPECS["persistent_q"].class_id, raw_juce)
    magic, _, _, loff = struct.unpack("<4si32sq", preset_bytes[:48])
    assert magic == b"VST3"
    assert preset_bytes[loff : loff + 4] == b"List"
    assert b"<VST3PluginState>" not in preset_bytes


def test_continuous_distance_envelopes_catch_subbass_quiet_drop() -> None:
    sr = 22050
    t_seg = np.linspace(0.0, 8.0, sr * 8, endpoint=False)
    # First 8s = Loud chorus (sub + dense 2.2kHz guitars); Second 8s = Quiet breakdown (sub STAYS LOUD, guitars DROP -8dB!)
    chorus = 0.25 * np.sin(2 * np.pi * 80 * t_seg) + 0.22 * np.sin(2 * np.pi * 2200 * t_seg)
    breakdown = 0.30 * np.sin(2 * np.pi * 80 * t_seg) + 0.05 * np.sin(2 * np.pi * 2200 * t_seg)
    inst_m = np.concatenate([chorus, breakdown]).astype(np.float32)
    inst_st = np.stack([inst_m, inst_m], axis=0)

    envs = compute_continuous_distance_envelopes(inst_st, sr=sr, anchor_span=(0, sr * 8), verse_intimacy_shift_db=0.3)
    # During breakdown (second half): gain & sat should pull down, room cohesion & soften should rise!
    assert float(np.mean(envs["gain_env"][sr * 10 : sr * 14])) < float(np.mean(envs["gain_env"][sr * 2 : sr * 6])) - 0.05
    assert float(np.mean(envs["sat_env"][sr * 10 : sr * 14])) < 0.70
    assert float(np.mean(envs["space_env"][sr * 10 : sr * 14])) > 1.12
    assert float(np.mean(envs["soften_env"][sr * 10 : sr * 14])) > 0.50


def test_true_stereo_harmony_leveling_and_flat_folder_discovery(tmp_path: Path) -> None:
    sr = 22050
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    lead = (0.20 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # Wide hard-panned opposite-phase-like stereo harmony (tiny Mid, large Side -> used to fool Mid-only leveling!)
    harm_st = np.stack([0.20 * np.sin(2 * np.pi * 554 * t), -0.18 * np.sin(2 * np.pi * 554 * t)], axis=0).astype(
        np.float32
    )
    out_h = process_dedicated_harmony_bus(harm_st, lead, sr=sr, target_rel_db=-11.2, side_boost_db=1.2)
    # Total stereo RMS of harmonies must stay clearly BELOW lead (~ -11 dB), NEVER booming near 0 dB!
    lead_rms = float(np.sqrt(np.mean(lead**2)))
    harm_stereo_rms = float(np.sqrt(0.5 * (np.mean(out_h[0] ** 2) + np.mean(out_h[1] ** 2))))
    rel_db = 20.0 * np.log10((harm_stereo_rms + 1e-12) / (lead_rms + 1e-12))
    assert -16.0 <= rel_db <= -8.5

    # Verify flat-folder discovery (Fall in Love layout without stems/ subfolder!)
    write_pcm24_wav(tmp_path / "20261125 - inst.wav", np.stack([lead, lead], axis=0), sr)
    write_pcm24_wav(tmp_path / "20261125 - main1.wav", lead, sr)
    write_pcm24_wav(tmp_path / "20261125 - harm1.wav", harm_st, sr)
    assets = discover_song_assets(tmp_path)
    assert assets["inst"] is not None and assets["inst"].name == "20261125 - inst.wav"
    assert len(assets["stems"]) == 2


def test_v6_precision_10band_eq_notch_bark_tamer_and_dual_harmony_split(tmp_path: Path) -> None:
    sr = 44100
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    rng = np.random.default_rng(99)

    # 1. Automated Narrow-Q Resonance Notch Detection on synthetic sharp 1.45 kHz ringing resonance
    base_noise = 0.08 * rng.standard_normal(len(t))
    ring_v = (base_noise + 0.28 * np.sin(2 * np.pi * 1450 * t)).astype(np.float32)
    notches = detect_vocal_surgical_notches(ring_v, sr=sr, max_notches=2)
    assert len(notches) >= 1
    assert any(1300.0 <= n[0] <= 1600.0 and n[1] <= -1.0 for n in notches)

    # 2. 10-Band Precision EQ zero-scoop anchor + backing-complementary dark-backing air guardrail
    dark_inst = (0.25 * np.sin(2 * np.pi * 220 * t) + 0.12 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    prof = compute_section_relative_profile(ring_v, np.stack([dark_inst, dark_inst], axis=0), sr=sr)
    eq10 = solve_precision_10band_eq_gpu(
        ring_v, np.stack([dark_inst, dark_inst], axis=0), sr=sr, target_profile=prof, inst_high_rolloff_db=-12.0
    )
    gains = eq10["precision_eq_gains_db"]
    assert isinstance(gains, dict) and len(gains) == 10
    assert gains["air_shelf"] <= 2.0  # Never blasts +5 dB air on dark backing!

    # 3. 6-Bark Consonant Transient Spike Tamer softens extreme 3.2 kHz spike without hurting vowel
    spiky = (0.14 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
    spiky[int(0.50 * sr) : int(0.515 * sr)] += (0.55 * np.sin(2 * np.pi * 3200 * t[int(0.50 * sr) : int(0.515 * sr)])).astype(np.float32)
    tamed = apply_bark_consonant_transient_tamer(spiky, sr=sr, amount=0.45)
    spike_before = float(np.max(np.abs(spiky[int(0.50 * sr) : int(0.515 * sr)])))
    spike_after = float(np.max(np.abs(tamed[int(0.50 * sr) : int(0.515 * sr)])))
    assert spike_after < spike_before * 0.90

    # 4. Separate Main Harmony vs Choir buses + audible Hollow-Core/Onset-Crushed harmony leveling
    p_m = tmp_path / "song - main1.wav"
    p_h = tmp_path / "song - +3.wav"
    p_c = tmp_path / "song - choir1.wav"
    write_pcm24_wav(p_m, (0.22 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), sr)
    write_pcm24_wav(p_h, (0.12 * np.sin(2 * np.pi * 415 * t)).astype(np.float32), sr)
    write_pcm24_wav(p_c, (0.10 * np.sin(2 * np.pi * 495 * t)).astype(np.float32), sr)
    cons = consolidate_vocal_stem_folder([p_m, p_h, p_c], target_sr=sr)
    assert cons.main_harmonies_stereo_bus is not None
    assert cons.choir_stereo_bus is not None
    mh = process_main_harmony_bus(cons.main_harmonies_stereo_bus, cons.lead_mono, sr=sr, target_rel_db=-8.6)
    rel_mh = 20.0 * np.log10(
        (float(np.sqrt(0.5 * (np.mean(mh[0] ** 2) + np.mean(mh[1] ** 2)))) + 1e-12)
        / (float(np.sqrt(np.mean(cons.lead_mono**2))) + 1e-12)
    )
    assert -12.5 <= rel_mh <= -5.5


def test_v7_lyrics_phoneme_alignment_and_raw_vocal_benchmark(tmp_path: Path) -> None:
    sr = 22050
    rng = np.random.default_rng(707)

    # Build a 3-phrase Japanese+English singing vocal with sharp unvoiced consonant attacks + steady harmonic vowels
    notes_hz = [261.626, 329.628, 391.995]  # C4, E4, G4 (C Major)
    phrase_dur = 0.90
    gap_dur = 0.45
    cons_dur = 0.08
    total_dur = 0.3 + len(notes_hz) * (phrase_dur + gap_dur)
    n_total = int(total_dur * sr)
    vocal = np.zeros(n_total, dtype=np.float32)

    sos_cons = signal.butter(4, [2800.0, 8200.0], btype="bandpass", fs=sr, output="sos")
    cursor = int(0.3 * sr)
    for f0_hz in notes_hz:
        # 80ms unvoiced consonant burst
        n_c = int(cons_dur * sr)
        raw_burst = rng.standard_normal(n_c + 200).astype(np.float64)
        burst = signal.sosfilt(sos_cons, raw_burst)[:n_c].astype(np.float32)
        burst /= float(np.max(np.abs(burst)) + 1e-6)
        burst *= 0.22 * np.hanning(n_c).astype(np.float32)
        vocal[cursor : cursor + n_c] = burst

        # Sustained harmonic vowel
        n_v = int((phrase_dur - cons_dur) * sr)
        t_v = np.arange(n_v, dtype=np.float64) / sr
        env = np.ones(n_v, dtype=np.float64)
        fade = int(0.025 * sr)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
        vowel = env * (
            0.24 * np.sin(2.0 * np.pi * f0_hz * t_v)
            + 0.12 * np.sin(2.0 * np.pi * 2.0 * f0_hz * t_v)
            + 0.05 * np.sin(2.0 * np.pi * 3.0 * f0_hz * t_v)
        )
        v_start = cursor + n_c
        vocal[v_start : v_start + n_v] = vowel.astype(np.float32)
        cursor += int((phrase_dur + gap_dur) * sr)

    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text(
        "[Verse]\n夜空に響く歌\n光る starlight を追いかけて\nWe shine tonight\n",
        encoding="utf-8",
    )

    alignment = align_lyrics_and_phonemes(vocal, sr=sr, lyrics_path=lyrics_file)
    assert alignment.method_used == "acoustic_dp"
    assert len(alignment.lines) == 3
    assert [ln.language for ln in alignment.lines] == ["ja", "mixed", "en"]
    assert all(ln.end_sec > ln.start_sec for ln in alignment.lines)
    assert np.any(alignment.vowel_core_mask)
    assert np.any(alignment.consonant_attack_protect_mask)

    snip = find_lyric_snippet_at_span(
        alignment, alignment.lines[1].start_sec + 0.05, alignment.lines[1].end_sec - 0.05
    )
    assert "starlight" in snip

    # Backwards-compatible 4-tuple & v7 5-tuple unpack on preprocess_vocal_pipeline
    inst_dummy = (0.08 * np.sin(2.0 * np.pi * 130.81 * np.arange(n_total) / sr)).astype(np.float32)
    prep_cfg = PreprocessConfig(enable_rx10_cleanup=False, target_deess_sib_gr_db=0.0, soothe_depth_db=0.0)
    v4, pt4, sc4, cal4 = preprocess_vocal_pipeline(vocal, inst_dummy, sr=sr, cfg=prep_cfg)
    assert len(v4) == len(vocal)
    assert pt4.median_f0_hz > 200.0 and len(sc4) == 3 and cal4.staged_lufs < -10.0

    _, _, _, _, ret_align = preprocess_vocal_pipeline(
        vocal, inst_dummy, sr=sr, cfg=prep_cfg, lyrics_path=lyrics_file, return_alignment=True
    )
    assert len(ret_align.lines) == 3

    # Raw vocal pitch correction + consonant attack preservation benchmark
    c_major_mask = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    bench = evaluate_raw_vocal_correction_benchmark(vocal, sr=sr, scale_mask=c_major_mask)
    assert bench["cents_improvement"] > 15.0
    assert bench["consonant_attack_band_preservation_corr"] >= 0.96


def test_v7_multiclass_fx_counter_melody_midi_and_s1_insert_slots(tmp_path: Path) -> None:
    sr = 22050
    dur = 14.0
    n = int(dur * sr)
    t = np.linspace(0.0, dur, n, endpoint=False)
    rng = np.random.default_rng(42)

    lead = np.zeros(n, dtype=np.float32)
    ref = np.zeros(n, dtype=np.float32)

    def _add_harm_tone(buf: np.ndarray, t0: float, t1: float, f0: float, amp: float = 0.22) -> None:
        s0, s1 = int(t0 * sr), int(t1 * sr)
        tt = np.arange(s1 - s0, dtype=np.float64) / sr
        sig = amp * (
            np.sin(2.0 * np.pi * f0 * tt)
            + 0.45 * np.sin(4.0 * np.pi * f0 * tt)
            + 0.25 * np.sin(6.0 * np.pi * f0 * tt)
            + 0.15 * np.sin(12.0 * np.pi * f0 * tt)
        )
        buf[s0:s1] = sig.astype(np.float32)

    # Phrase 1 (0.4-1.2s): spot_delay_throw (1/8 note echoes @ 120 BPM = 0.25s spacing during 1.6s rest)
    _add_harm_tone(lead, 0.4, 1.2, 329.63)
    _add_harm_tone(ref, 0.4, 1.0, 329.63)
    p1_tail = np.zeros(n, dtype=np.float32)
    _add_harm_tone(p1_tail, 1.04, 1.20, 329.63)
    ref += Pedalboard([Delay(delay_seconds=0.25, feedback=0.62, mix=0.52)])(p1_tail[None, :], sr)[0]
    ref += (0.003 * rng.standard_normal(n)).astype(np.float32)

    # Phrase 2 (2.8-4.2s): radio_voice (strictly band-limited 1500 Hz mid tone)
    _add_harm_tone(lead, 2.8, 4.2, 329.63)
    r_s0, r_s1 = int(2.8 * sr), int(4.2 * sr)
    ref[r_s0:r_s1] = (0.24 * np.sin(2.0 * np.pi * 1550.0 * t[r_s0:r_s1])).astype(np.float32)

    # Phrase 3 (4.8-6.1s): distortion_grit (keep 220Hz & 6kHz so not radio, inject dense 1.2-4.2kHz grit)
    _add_harm_tone(lead, 4.8, 6.1, 261.63)
    g_s0, g_s1 = int(4.8 * sr), int(6.1 * sr)
    sos_grit = signal.butter(4, [1300.0, 4000.0], btype="bandpass", fs=sr, output="sos")
    grit_noise = signal.sosfilt(sos_grit, rng.standard_normal(g_s1 - g_s0 + 200))[: g_s1 - g_s0]
    grit_noise = (grit_noise / (np.max(np.abs(grit_noise)) + 1e-6)) * 0.28
    ref[g_s0:g_s1] = (
        0.09 * np.sin(2.0 * np.pi * 220.0 * t[g_s0:g_s1])
        + 0.05 * np.sin(2.0 * np.pi * 6200.0 * t[g_s0:g_s1])
        + grit_noise
    ).astype(np.float32)

    # Phrase 4 (6.6-8.3s): hard_autotune (ref flat steps C4->D4->E4->F4, lead natural 5.5Hz vibrato)
    at_s0, at_s1 = int(6.6 * sr), int(8.3 * sr)
    tt_at = np.arange(at_s1 - at_s0, dtype=np.float64) / sr
    vib_hz = 293.66 * (2.0 ** (0.42 * np.sin(2.0 * np.pi * 5.5 * tt_at) / 12.0))
    phase_u = 2.0 * np.pi * np.cumsum(vib_hz) / sr
    lead[at_s0:at_s1] = (0.22 * np.sin(phase_u) + 0.08 * np.sin(2.0 * phase_u)).astype(np.float32)
    for k_s, f_step in enumerate((261.63, 293.66, 329.63, 349.23)):
        st0 = at_s0 + int(k_s * 0.42 * sr)
        st1 = min(at_s1, st0 + int(0.42 * sr))
        tt_s = np.arange(st1 - st0, dtype=np.float64) / sr
        ref[st0:st1] = (0.22 * np.sin(2.0 * np.pi * f_step * tt_s) + 0.08 * np.sin(4.0 * np.pi * f_step * tt_s)).astype(
            np.float32
        )

    # Phrase 5 (9.8-11.2s) & Phrase 6 (12.6-13.6s): normal phrases for baseline + Chorus rest counter-melody
    _add_harm_tone(lead, 9.8, 11.2, 261.63)
    _add_harm_tone(ref, 9.8, 11.2, 261.63)
    _add_harm_tone(lead, 12.6, 13.6, 392.00)
    _add_harm_tone(ref, 12.6, 13.6, 392.00)

    inst = (
        0.18 * np.sin(2.0 * np.pi * 261.63 * t)
        + 0.14 * np.sin(2.0 * np.pi * 329.63 * t)
        + 0.14 * np.sin(2.0 * np.pi * 392.00 * t)
    ).astype(np.float32)

    events = detect_creative_fx_events(lead, inst, sr=sr, ref_vocal_mono=ref, bpm=120.0)
    detected_types = {ev.fx_type for ev in events}
    assert {"spot_delay_throw", "radio_voice", "distortion_grit", "hard_autotune"}.issubset(detected_types)

    lead_fx, spot_stems = apply_verified_inline_fx_chain(lead, sr=sr, events=events)
    assert len(lead_fx) == len(lead)
    assert {"radio_voice", "distortion_grit", "hard_autotune"}.issubset(set(spot_stems.keys()))

    spot_dly_st = render_spot_delay_throw_stem(lead_fx, sr=sr, events=events, bpm=120.0)
    assert spot_dly_st.shape == (2, n) and float(np.max(np.abs(spot_dly_st))) > 1e-3

    # Stage 4 Counter-melody & 4-track MIDI export
    sections = [
        SongSection(0, "01_Verse_PreChorus", 0.0, 8.8, False, -14.0, -0.5, None),
        SongSection(1, "02_Chorus", 8.8, 14.0, True, -10.0, 0.0, None),
    ]
    midi_out, counter_notes = generate_counter_melody_and_export_midi(
        tmp_path / "Director_Guide.mid",
        bpm=120.0,
        lead_vocal_mono=lead,
        inst_mono=inst,
        sections=sections,
        sr=sr,
    )
    assert midi_out.exists() and len(counter_notes) >= 1
    mid = mido.MidiFile(str(midi_out))
    assert len(mid.tracks) == 4
    assert sum(1 for tr in mid.tracks for msg in tr if msg.type == "note_on") >= 4

    # Studio One .song build with insert_slots, fine_main_events, tempo_segments & .envelopex
    template_path = Path(r"C:\MIX_automation\data\templates\test20261124\test20261124.song")
    if not template_path.exists():
        return

    main_wav = tmp_path / "main.wav"
    fx_wav = tmp_path / "fx_radio.wav"
    write_pcm24_wav(main_wav, lead_fx, sr)
    write_pcm24_wav(fx_wav, np.stack([spot_stems["radio_voice"], spot_stems["radio_voice"]], axis=0), sr)

    vs2_pre_rel = "Presets/Channels/fx_radio_vocalsynth2/1 - VocalSynth 2.vstpreset"
    presets = {
        vs2_pre_rel: create_vstpreset_bytes(DEFAULT_S1_PLUGIN_SPECS["vocalsynth2"].class_id, b"VS2_RADIO"),
    }
    extra_trks = [
        TrackMediaSpec(
            name="fx_radio_vocalsynth2",
            wav_path=fx_wav,
            stereo=True,
            gain_db=-1.0,
            mute=False,
            insert_slots=(("vocalsynth2", vs2_pre_rel),),
        )
    ]
    fine_ev = [(0.0, 4.4, -0.8, "01_Verse_part1"), (4.4, 8.8, +0.2, "01_Verse_part2"), (8.8, 14.0, 0.0, "02_Chorus")]
    tempo_segs = [(0.0, 120.0), (16.0, 132.0)]

    out_song = build_studio_one_song(
        template_song_path=template_path,
        output_song_dir=tmp_path,
        song_name="V7Stage3S1Check",
        bpm=120.0,
        sr=sr,
        main_vocal_wav=main_wav,
        sections=sections,
        extra_tracks=extra_trks,
        patched_presets=presets,
        fine_main_events=fine_ev,
        tempo_segments=tempo_segs,
    )
    with zipfile.ZipFile(out_song, "r") as zf:
        names = set(zf.namelist())
        song_xml = zf.read("Song/song.xml").decode("utf-8")
        mixer_xml = zf.read("Devices/audiomixer.xml").decode("utf-8")
        assert all(b"<VST3PluginState>" not in zf.read(nm) for nm in names)

    assert "Envelopes/fx_radio_vocalsynth2/ボリューム.envelopex" in names
    assert "Envelopes/fx_radio_vocalsynth2/パン.envelopex" in names
    assert vs2_pre_rel in names
    assert "VocalSynth 2" in mixer_xml and vs2_pre_rel in mixer_xml

    song_root = ET.fromstring(song_xml.lstrip("\ufeff").replace("<Song ", '<Song xmlns:x="http://www.presonus.com/xml" ', 1))
    assert len(song_root.findall(".//TempoMapSegment")) == 2
    main_evs = [ev for m in song_root.iter("MediaTrack") if m.attrib.get("name") == "mainvocal" for ev in m.iter("AudioEvent")]
    assert len(main_evs) == 3


def test_v7_multisection_sculpting_tempo_map_preference_learner_and_ref_gate(tmp_path: Path) -> None:
    sr = 22050
    # 1. Multi-section adaptive sculpting (top_edge_6500 <= 0.0 across all sections!)
    t_sec = np.linspace(0.0, 6.0, sr * 6, endpoint=False)
    v_seg = (
        0.22 * np.sin(2.0 * np.pi * 240.0 * t_sec)
        + 0.14 * np.sin(2.0 * np.pi * 2600.0 * t_sec)
        + 0.09 * np.sin(2.0 * np.pi * 6800.0 * t_sec)
    ).astype(np.float32)
    v_full = np.concatenate([v_seg * 0.9, v_seg * 0.85, v_seg]).astype(np.float32)

    i_verse = (0.16 * np.sin(2.0 * np.pi * 220.0 * t_sec) + 0.06 * np.sin(2.0 * np.pi * 2200.0 * t_sec)).astype(np.float32)
    i_quiet = (0.06 * np.sin(2.0 * np.pi * 110.0 * t_sec)).astype(np.float32)
    i_chorus = (0.32 * np.sin(2.0 * np.pi * 220.0 * t_sec) + 0.25 * np.sin(2.0 * np.pi * 2400.0 * t_sec)).astype(np.float32)
    i_full = np.concatenate([i_verse, i_quiet, i_chorus]).astype(np.float32)
    i_full_st = np.stack([i_full, i_full], axis=0)

    raw_spans = [
        (0.0, 6.0, "01_Verse_PreChorus"),
        (6.0, 12.0, "02_Quiet_Bridge_Verse"),
        (12.0, 18.0, "03_Chorus"),
    ]
    tgt_prof = compute_section_relative_profile(v_seg, np.stack([i_chorus, i_chorus], axis=0), sr=sr)
    v_adapt, diag = apply_multisection_adaptive_sculpting(
        v_full, i_full_st, sr=sr, raw_spans=raw_spans, anchor_idx=2, target_profile=tgt_prof, verse_intimacy_shift_db=0.5
    )
    assert v_adapt.shape == v_full.shape and len(diag) == 3
    assert all(d["band_gains_db"]["top_edge_6500"] <= 0.0 for d in diag)

    # 2. Continuous Tempo Map (steady vs variable BPM)
    def _click_track(bpm_list: list[tuple[float, float]]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        click_len = int(0.025 * sr)
        tt_c = np.arange(click_len, dtype=np.float32) / sr
        pulse = (0.75 * np.sin(2.0 * np.pi * 1200.0 * tt_c) * np.exp(-tt_c * 140.0)).astype(np.float32)
        for seg_dur, b_val in bpm_list:
            buf = np.zeros(int(seg_dur * sr), dtype=np.float32)
            step_s = int((60.0 / b_val) * sr)
            for pos in range(0, len(buf) - click_len, step_s):
                buf[pos : pos + click_len] += pulse
            chunks.append(buf)
        return np.concatenate(chunks)

    steady_info = estimate_continuous_tempo_map(_click_track([(14.0, 128.0)]), sr=sr)
    assert not steady_info["is_variable_tempo"]
    assert 125.0 <= steady_info["global_bpm"] <= 131.0

    var_info = estimate_continuous_tempo_map(_click_track([(12.0, 100.0), (12.0, 138.0)]), sr=sr)
    assert var_info["is_variable_tempo"]
    assert len(var_info["tempo_segments"]) >= 2

    # 3. Preference Learner + Advanced Anchor Propagation
    template_path = Path(r"C:\MIX_automation\data\templates\test20261124\test20261124.song")
    if template_path.exists():
        pref_dir = tmp_path / "pref_songs"
        pref_dir.mkdir()
        m_wav = pref_dir / "main.wav"
        h_wav = pref_dir / "harm.wav"
        write_pcm24_wav(m_wav, v_full[: sr * 12], sr)
        write_pcm24_wav(h_wav, v_full[: sr * 12] * 0.5, sr)

        pq_xml_a = b'<PersistantQState><Band id="3" enabled="1" type="0" frequencyHz="420.0" gainDb="-1.0" dynamicEnabled="0" spectralEnabled="0"/></PersistantQState>\x00'
        pq_xml_u = b'<PersistantQState><Band id="3" enabled="1" type="0" frequencyHz="420.0" gainDb="+0.5" dynamicEnabled="0" spectralEnabled="0"/></PersistantQState>\x00'
        secs_p = [
            SongSection(0, "01_Verse", 0.0, 6.0, False, -16.0, -1.0, None),
            SongSection(1, "02_Chorus", 6.0, 12.0, True, -10.0, 0.0, None),
        ]
        chain_p = VocalChainParams(
            archetype_chain="dense_rock_edm",
            selected_plugins=["Persistent Q"],
            hpf_hz=90.0,
            body_gain_db=0.0,
            mud_cut_db=-1.0,
            core_mid_gain_db=0.0,
            presence_gain_db=0.5,
            air_shelf_db=0.0,
            vca_threshold_db=-22.0,
            vca_measured_gr_db=2.0,
            comp2_param=38.0,
            comp2_measured_gr_db=2.0,
            parallel_sat_amount=1.2,
            parallel_sat_wet_pct=10.0,
            vocal_fader_db=0.0,
            early_room_send_db=-18.0,
            slap_delay_send_db=-19.0,
            tempo_delay_send_db=-20.0,
            plate_reverb_send_db=-19.0,
            doubler_send_db=-22.0,
            unmask_mid_dip_db=1.0,
        )
        auto_song = build_studio_one_song(
            template_song_path=template_path,
            output_song_dir=pref_dir,
            song_name="AutoVer",
            bpm=120.0,
            sr=sr,
            main_vocal_wav=m_wav,
            harm1_wav=h_wav,
            sections=secs_p,
            extra_tracks=[],
            patched_presets={
                "Presets/Channels/mainvocal/1 - Persistent Q.vstpreset": create_vstpreset_bytes(
                    DEFAULT_S1_PLUGIN_SPECS["persistent_q"].class_id, pq_xml_a
                )
            },
            chain_params=chain_p,
        )
        (pref_dir / "mix_report.json").write_text(
            json.dumps(
                {
                    "optimized_chain_params": chain_p.to_dict(),
                    "sections": [
                        {"index": 0, "label": "01_Verse", "start_sec": 0.0, "end_sec": 6.0, "is_anchor": False, "inst_rms_db": -16.0, "clip_gain_db": -1.0},
                        {"index": 1, "label": "02_Chorus", "start_sec": 6.0, "end_sec": 12.0, "is_anchor": True, "inst_rms_db": -10.0, "clip_gain_db": 0.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        user_song = build_studio_one_song(
            template_song_path=template_path,
            output_song_dir=pref_dir,
            song_name="UserVer",
            bpm=120.0,
            sr=sr,
            main_vocal_wav=m_wav,
            harm1_wav=h_wav,
            sections=[
                SongSection(0, "01_Verse", 0.0, 6.0, False, -16.0, -1.0, None),
                SongSection(1, "02_Chorus", 6.0, 12.0, True, -10.0, +1.5, None),
            ],
            extra_tracks=[],
            patched_presets={
                "Presets/Channels/mainvocal/1 - Persistent Q.vstpreset": create_vstpreset_bytes(
                    DEFAULT_S1_PLUGIN_SPECS["persistent_q"].class_id, pq_xml_u
                )
            },
            chain_params=VocalChainParams(**{**chain_p.to_dict(), "vocal_fader_db": +1.2}),
        )
        offsets_json = tmp_path / "user_preference_offsets.json"
        rec = update_user_preference_offsets(auto_song, user_song, offsets_json, alpha=0.5)
        assert rec["num_sessions"] == 1
        assert rec["k_lufs_vi_offset_db"] > 0.5
        assert rec["intrinsic_band_offsets_db"]["boxiness_mud"] > 0.3

        shifted_prof = apply_user_preference_offsets(tgt_prof, offsets_json)
        assert "+pref" in shifted_prof.backing_archetype
        assert shifted_prof.k_lufs_vi_ratio_db > tgt_prof.k_lufs_vi_ratio_db

        prop_song, prop_sum = propagate_anchor_advanced(user_song, anchor_event_index=1)
        assert prop_song.exists()
        assert prop_sum["harmony_gain_delta_db"] > 1.0
        assert prop_sum["updated_events"][0]["delta_db"] > 0.5

    # 4. Reference Candidate Quality Gates (44.1 kHz bandwidth, stereo, duplicate ID & key-shift)
    sr_ref = 44100
    t_r = np.linspace(0.0, 4.0, sr_ref * 4, endpoint=False)
    rng_r = np.random.default_rng(11)
    env_r = (0.4 + 0.5 * np.sin(2.0 * np.pi * 1.5 * t_r) ** 2).astype(np.float32)
    base_chord = (
        0.22 * np.sin(2.0 * np.pi * 261.63 * t_r)
        + 0.18 * np.sin(2.0 * np.pi * 329.63 * t_r)
        + 0.16 * np.sin(2.0 * np.pi * 392.00 * t_r)
        + 0.08 * rng_r.standard_normal(len(t_r))
    ).astype(np.float32) * env_r
    wide_st = np.stack([base_chord * 1.05, np.roll(base_chord, 12) * 0.95], axis=0).astype(np.float32)

    good_dir = tmp_path / "artist" / "good_song"
    (good_dir / "offvocal").mkdir(parents=True)
    write_pcm24_wav(good_dir / "cover_GOOD01.wav", wide_st, sr_ref)
    write_pcm24_wav(good_dir / "offvocal" / "inst.wav", wide_st, sr_ref)
    good_res = validate_reference_song_candidate(good_dir, existing_cover_ids={"OTHER99"}, target_sr=sr_ref)
    assert good_res["passed"]
    assert not good_res["key_shifted"] and good_res["paired_calibration_eligible"]

    # Brickwall LPF 11 kHz + mono-collapsed + duplicate cover ID
    sos_lp = signal.butter(6, 11000.0, btype="lowpass", fs=sr_ref, output="sos")
    muff_m = signal.sosfilt(sos_lp, base_chord.astype(np.float64)).astype(np.float32)
    bad_dir = tmp_path / "artist" / "bad_song"
    bad_dir.mkdir(parents=True)
    write_pcm24_wav(bad_dir / "cover_GOOD01.wav", np.stack([muff_m, muff_m], axis=0), sr_ref)
    bad_res = validate_reference_song_candidate(bad_dir, existing_cover_ids={"GOOD01"}, target_sr=sr_ref)
    assert not bad_res["passed"] and bad_res["is_duplicate"]
    assert any("brickwall" in r for r in bad_res["rejection_reasons"])
    assert any("mono_collapsed" in r for r in bad_res["rejection_reasons"])

    # Key-shifted off-vocal (+2 semitones: C-E-G vs D-F#-A)
    ks_dir = tmp_path / "artist" / "shifted_song"
    (ks_dir / "offvocal").mkdir(parents=True)
    shifted_chord = (
        0.22 * np.sin(2.0 * np.pi * 293.66 * t_r)
        + 0.18 * np.sin(2.0 * np.pi * 369.99 * t_r)
        + 0.16 * np.sin(2.0 * np.pi * 440.00 * t_r)
        + 0.08 * rng_r.standard_normal(len(t_r))
    ).astype(np.float32) * env_r
    shifted_st = np.stack([shifted_chord * 1.05, np.roll(shifted_chord, 12) * 0.95], axis=0).astype(np.float32)
    write_pcm24_wav(ks_dir / "cover_SHIFT01.wav", wide_st, sr_ref)
    write_pcm24_wav(ks_dir / "offvocal" / "inst.wav", shifted_st, sr_ref)
    ks_res = validate_reference_song_candidate(ks_dir, target_sr=sr_ref)
    assert ks_res["key_shifted"]
    assert abs(ks_res["key_shift_semitones"]) == 2
    assert not ks_res["paired_calibration_eligible"]


def test_v7_1_studio_one_sync_lock_coactive_choir_and_smart_asset_discovery(tmp_path: Path) -> None:
    sr = 22050
    dur = 8.0
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)

    # 1. Smart asset discovery: prefers `(修正版)` healthy off-vocal over -38dBFS unmastered stem & ignores Mixdown.wav!
    raw_dir = tmp_path / "eva_like"
    raw_dir.mkdir()
    lead_sig = (0.20 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    quiet_off = np.stack([lead_sig * 0.02, lead_sig * 0.02], axis=0)
    fixed_off = np.stack([lead_sig * 0.90, lead_sig * 0.90], axis=0)
    write_pcm24_wav(raw_dir / "song - main1.wav", lead_sig, sr)
    write_pcm24_wav(raw_dir / "song - offvocal.wav", quiet_off, sr)
    write_pcm24_wav(raw_dir / "song - offvocal(修正版).wav", fixed_off, sr)
    write_pcm24_wav(raw_dir / "20261126Mixdown.wav", fixed_off, sr)

    assets = discover_song_assets(raw_dir)
    assert assets["inst"] is not None and "修正版" in assets["inst"].name
    assert assets["original"] is None  # Diagnostic Mixdown.wav must NEVER be mistaken for original reference!

    # 2. Co-active Choir vs Main Harmony leveling:
    # Lead is quiet (-22 dBFS) in first 4s verse and loud (-10 dBFS) in last 4s chorus; Choir ONLY sings in chorus!
    lead_dyn = np.concatenate([lead_sig[: n // 2] * 0.25, lead_sig[n // 2 :] * 1.4]).astype(np.float32)
    choir_sparse = np.zeros((2, n), dtype=np.float32)
    choir_sparse[0, n // 2 :] = 0.08 * np.sin(2.0 * np.pi * 523.25 * t[n // 2 :])
    choir_sparse[1, n // 2 :] = 0.08 * np.sin(2.0 * np.pi * 659.25 * t[n // 2 :])

    choir_out = process_choir_bus(choir_sparse, lead_dyn, sr=sr, target_rel_db=-7.8)
    chorus_lead_rms = float(np.sqrt(np.mean(lead_dyn[n // 2 :] ** 2)))
    chorus_choir_rms = float(np.sqrt(0.5 * (np.mean(choir_out[0, n // 2 :] ** 2) + np.mean(choir_out[1, n // 2 :] ** 2))))
    choir_rel_chorus_db = 20.0 * np.log10((chorus_choir_rms + 1e-12) / (chorus_lead_rms + 1e-12))
    assert -10.5 <= choir_rel_chorus_db <= -5.5  # Audible in chorus, NOT dragged down by quiet verse!

    # Main harmony sings constant amplitude across BOTH quiet verse and loud chorus -> lead contour pocket tucks verse down!
    harm_const = np.stack(
        [0.15 * np.sin(2.0 * np.pi * 415.3 * t), 0.15 * np.sin(2.0 * np.pi * 415.3 * t)],
        axis=0,
    ).astype(np.float32)
    harm_out = process_main_harmony_bus(harm_const, lead_dyn, sr=sr, target_rel_db=-11.2)
    verse_h_rms = float(np.sqrt(0.5 * (np.mean(harm_out[0, sr : 3 * sr] ** 2) + np.mean(harm_out[1, sr : 3 * sr] ** 2))))
    chorus_h_rms = float(np.sqrt(0.5 * (np.mean(harm_out[0, 5 * sr : 7 * sr] ** 2) + np.mean(harm_out[1, 5 * sr : 7 * sr] ** 2))))
    assert verse_h_rms < chorus_h_rms * 0.55

    # 3. Studio One .song variable tempo segments MUST lock `tempoFollow="0"` and `start == offset` (zero vocal slip!)
    template_path = Path(r"C:\MIX_automation\data\templates\test20261124\test20261124.song")
    if template_path.exists():
        s1_dir = tmp_path / "s1_sync"
        s1_dir.mkdir()
        mv_wav = s1_dir / "mv.wav"
        write_pcm24_wav(mv_wav, lead_dyn, sr)
        out_song = build_studio_one_song(
            template_song_path=template_path,
            output_song_dir=s1_dir,
            song_name="SyncLockTest",
            bpm=140.0,
            sr=sr,
            main_vocal_wav=mv_wav,
            sections=[],
            extra_tracks=[TrackMediaSpec(name="offvocal_unmasked", wav_path=mv_wav, stereo=False)],
            fine_main_events=[(0.0, 2.5, 0.0, "ev1"), (2.5, 5.5, -0.5, "ev2"), (5.5, 8.0, +0.4, "ev3")],
            tempo_segments=[(0.0, 140.0), (4.0, 112.0), (8.0, 140.0)],
        )
        with zipfile.ZipFile(out_song, "r") as zf:
            s_xml = zf.read("Song/song.xml").decode("utf-8").lstrip("\ufeff")
        s_root = ET.fromstring(s_xml.replace("<Song ", '<Song xmlns:x="http://www.presonus.com/xml" ', 1))
        for mtrk in s_root.iter("MediaTrack"):
            if mtrk.attrib.get("mediaType") == "Audio":
                assert mtrk.attrib.get("tempoFollow") == "0"
            for aev in mtrk.iter("AudioEvent"):
                if "start" in aev.attrib and "offset" in aev.attrib:
                    assert abs(float(aev.attrib["start"]) - float(aev.attrib["offset"])) < 1e-6
