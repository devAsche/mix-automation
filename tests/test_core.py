"""v2 Unit tests for acoustic profiling, Stage 3 evidence gating, chord-safe harmonies & Studio One .song XML."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from mix_automation.analysis.relative_profile import (
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
    detect_song_sections,
    solve_differentiable_intrinsic_eq_gpu,
)
from mix_automation.io.audio_loader import consolidate_vocal_stem_folder
from mix_automation.io.song_builder import (
    build_studio_one_song,
    patch_vstpreset_component_state,
    write_pcm24_wav,
)
from mix_automation.io.studio_one import write_vstpreset


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
