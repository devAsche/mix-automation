"""Unit tests for relative acoustic profiling, anchor propagation, and Studio One 5 .vstpreset patching."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from mix_automation.analysis.relative_profile import (
    compute_anchor_propagation,
    compute_section_relative_profile,
)
from mix_automation.dsp.vst3_optimizer import detect_song_sections
from mix_automation.io.song_builder import patch_vstpreset_component_state
from mix_automation.io.studio_one import write_vstpreset


def test_relative_profile_and_anchor_propagation() -> None:
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)

    vocal_chorus = 0.25 * np.sin(2 * np.pi * 260 * t) + 0.15 * np.sin(2 * np.pi * 2400 * t)
    inst_chorus = 0.30 * np.sin(2 * np.pi * 120 * t) + 0.25 * np.sin(2 * np.pi * 2200 * t)
    anchor = compute_section_relative_profile(vocal_chorus, inst_chorus, sr=sr)

    inst_verse = inst_chorus * 0.25
    raw_verse = compute_section_relative_profile(vocal_chorus, inst_verse, sr=sr)

    adjustment = compute_anchor_propagation(anchor, raw_verse)
    assert -13.0 <= adjustment.input_gain_delta_db <= -11.0


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


def test_detect_song_sections_identifies_loudest_chorus() -> None:
    sr = 22050
    # 18s quiet verse followed by 18s loud chorus
    t_seg = np.linspace(0.0, 18.0, sr * 18, endpoint=False)
    verse_i = 0.08 * np.sin(2 * np.pi * 220 * t_seg)
    chorus_i = 0.45 * np.sin(2 * np.pi * 220 * t_seg) + 0.25 * np.sin(2 * np.pi * 1760 * t_seg)
    inst = np.concatenate([verse_i, chorus_i]).astype(np.float32)
    vocal = (0.20 * np.sin(2 * np.pi * 330 * np.linspace(0.0, 36.0, sr * 36, endpoint=False))).astype(np.float32)

    sections, anchor_idx = detect_song_sections(vocal, inst, sr=sr, min_section_sec=10.0)
    assert len(sections) >= 2
    a_start, a_end, _ = sections[anchor_idx]
    # Anchor should cover the loud second half
    assert a_end > 25.0
