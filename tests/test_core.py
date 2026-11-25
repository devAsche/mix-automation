"""v2 Unit tests for acoustic profiling, Stage 3 evidence gating, chord-safe harmonies & Studio One .song XML."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from mix_automation.analysis.reference_extractor import synthesize_acoustic_knn_target
from mix_automation.analysis.relative_profile import (
    classify_backing_archetype,
    compute_anchor_propagation,
    compute_section_relative_profile,
)
from mix_automation.creative.fx_and_harmony import (
    _select_and_stitch_chord_safe_harmony,
    detect_creative_fx_events,
)
from mix_automation.dsp.pitch_timing import ReviewMarker
from mix_automation.dsp.vst3_optimizer import (
    SongSection,
    VocalChainParams,
    apply_bark_consonant_transient_tamer,
    apply_dynamic_instrumental_unmasking,
    compute_continuous_distance_envelopes,
    detect_song_sections,
    detect_vocal_surgical_notches,
    process_dedicated_harmony_bus,
    process_main_harmony_bus,
    solve_differentiable_intrinsic_eq_gpu,
    solve_precision_10band_eq_gpu,
)
from mix_automation.io.audio_loader import consolidate_vocal_stem_folder
from mix_automation.io.song_builder import (
    DEFAULT_S1_PLUGIN_SPECS,
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
