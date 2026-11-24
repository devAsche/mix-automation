"""Unit tests for relative acoustic profiling, anchor propagation, and .vstpreset I/O."""

from __future__ import annotations

from pathlib import Path
import struct
import numpy as np

from mix_automation.analysis.relative_profile import (
    compute_anchor_propagation,
    compute_section_relative_profile,
)
from mix_automation.io.studio_one import write_vstpreset


def test_relative_profile_and_anchor_propagation() -> None:
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)

    # Synthetic vocal (fundamental + presence harmonics) and backing track
    vocal_chorus = 0.25 * np.sin(2 * np.pi * 260 * t) + 0.15 * np.sin(2 * np.pi * 2400 * t)
    inst_chorus = 0.30 * np.sin(2 * np.pi * 120 * t) + 0.25 * np.sin(2 * np.pi * 2200 * t)

    anchor = compute_section_relative_profile(vocal_chorus, inst_chorus, sr=sr)

    # Verse where backing track drops by ~12 dB (amplitude x0.25), while raw vocal stays loud
    inst_verse = inst_chorus * 0.25
    raw_verse = compute_section_relative_profile(vocal_chorus, inst_verse, sr=sr)

    adjustment = compute_anchor_propagation(anchor, raw_verse)
    # Since backing dropped ~12 dB while vocal didn't, vocal gain delta must be negative (~ -12 dB)
    assert -13.0 <= adjustment.input_gain_delta_db <= -11.0


def test_write_vstpreset_binary_layout(tmp_path: Path) -> None:
    dummy_cid = "0123456789ABCDEF0123456789ABCDEF"
    dummy_state = b"<VST3State value='42'/>\x00"
    out = write_vstpreset(tmp_path / "test.vstpreset", dummy_cid, dummy_state)

    raw = out.read_bytes()
    magic, version, cid_bytes, list_offset = struct.unpack("<4si32sq", raw[:48])
    assert magic == b"VST3"
    assert version == 1
    assert cid_bytes.decode("ascii") == dummy_cid
    assert list_offset == 48 + len(dummy_state)
    assert raw[48:list_offset] == dummy_state
    assert raw[list_offset : list_offset + 4] == b"List"
