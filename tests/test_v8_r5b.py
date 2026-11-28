"""R5b: 修正後 stem preference, correction EQ source precedence (explicit > song index > voice > auto),
export_song picking up correction_eq_applied.json."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mix_automation.pipeline import discover_song_assets, prefer_revised_stems
from mix_automation.v8 import correction_eq as ceq
from mix_automation.v8 import engine

ROOT = Path(__file__).resolve().parents[1]
EQ = {"format": ceq.FORMAT, "voice": "t", "bands": [{"band": 1, "type": "Peaking", "frequency_hz": 3000.0,
                                                     "gain_db": -2.0, "q": 1.0}]}


def _wav(p: Path) -> Path:
    sf.write(str(p), np.zeros(441, dtype=np.float32), 44100)
    return p


def test_prefer_revised_stems_keeps_unmatched():
    files = [Path(n) for n in ("エバ - edエバ - main 1.wav", "エバ - edエバ - main 2.wav", "エバ - harmo.wav",
                               "エバ修正後 - edエバ - main 1.wav", "エバ修正後  - edエバ - main 2.wav")]
    kept, sup = prefer_revised_stems(files)
    assert [p.name for p in sup] == ["エバ - edエバ - main 1.wav", "エバ - edエバ - main 2.wav"]
    assert {p.name for p in kept} == {"エバ - harmo.wav", "エバ修正後 - edエバ - main 1.wav",
                                      "エバ修正後  - edエバ - main 2.wav"}


def test_discover_flat_folder_prefers_revised(tmp_path):
    for i in (1, 2, 3):
        _wav(tmp_path / f"エバ - edエバ - main {i}.wav")
        _wav(tmp_path / f"エバ修正後 - edエバ - main {i}.wav")
    _wav(tmp_path / "エバ - offvocal.wav")
    a = discover_song_assets(tmp_path)
    assert sorted(p.name for p in a["stems"]) == [f"エバ修正後 - edエバ - main {i}.wav" for i in (1, 2, 3)]
    assert len(a["stems_superseded"]) == 3
    assert all(p.exists() for p in a["stems_superseded"])  # never deleted


def test_discover_stems_subfolder_prefers_revised(tmp_path):
    (tmp_path / "stems").mkdir()
    _wav(tmp_path / "stems" / "main.wav")
    _wav(tmp_path / "stems" / "main修正後.wav")
    _wav(tmp_path / "stems" / "harmo.wav")
    _wav(tmp_path / "song inst.wav")
    a = discover_song_assets(tmp_path)
    assert sorted(p.name for p in a["stems"]) == ["harmo.wav", "main修正後.wav"]
    assert [p.name for p in a["stems_superseded"]] == ["main.wav"]


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "エバ").exists(), reason="real data not present")
def test_discover_real_eva_keeps_three_revised():
    a = discover_song_assets(ROOT / "data" / "raw" / "エバ")
    assert len(a["stems"]) == 3 and all("修正後" in p.name for p in a["stems"])
    assert len(a["stems_superseded"]) == 3


def _profiles(tmp_path: Path) -> Path:
    ceq.save(EQ, tmp_path / "song_a.json")
    (tmp_path / "index.json").write_text(json.dumps({"format": ceq.INDEX_FORMAT, "songs": {"A": "song_a.json"}}),
                                         encoding="utf-8")
    return tmp_path / "index.json"


def test_resolve_source_precedence(tmp_path, monkeypatch):
    idx = _profiles(tmp_path)
    explicit = ceq.save({**EQ, "voice": "x"}, tmp_path / "x.json")
    src, p, cfg = ceq.resolve_source("A", "nobody", explicit, idx)
    assert (src, p, cfg["voice"]) == ("explicit", explicit, "x")
    src, p, cfg = ceq.resolve_source("A", "nobody", None, idx)
    assert (src, p.name) == ("song_index", "song_a.json") and not ceq.is_flat(cfg)
    monkeypatch.setattr(ceq, "PROFILE_DIR", tmp_path)
    ceq.save({**EQ, "voice": "v"}, tmp_path / "v.json")
    src, p, cfg = ceq.resolve_source("B", "v", None, idx)
    assert (src, cfg["voice"]) == ("voice", "v")
    assert ceq.resolve_source("B", "nobody", None, idx) == ("auto", None, None)
    assert ceq.resolve_source("B", "nobody", None, tmp_path / "missing_index.json") == ("auto", None, None)


def test_song_index_missing_file_raises(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({"format": ceq.INDEX_FORMAT, "songs": {"A": "gone.json"}}),
                                         encoding="utf-8")
    with pytest.raises(ceq.CorrectionEQError):
        ceq.song_index_path("A", tmp_path / "index.json")


def test_repo_index_entries_exist_and_validate():
    for song in ("エバ", "ラブ&デストロイ"):
        p = ceq.song_index_path(song)
        assert p is not None and ceq.load(p) is not None


def test_select_correction_eq_auto(monkeypatch):
    seen = {}
    broad = {"hz": 10000.0, "gain_db": 1.5}

    def fake(x, sr, voice="ryo", gap_db=None):
        seen.update(n=len(x), sr=sr, voice=voice, gap=gap_db)
        return {**EQ, "source": {"broad_high": broad}}

    def fake_gap(v, i):
        seen["gap_args"] = (len(v), np.ndim(i))
        return 2.345

    monkeypatch.setitem(sys.modules, "mix_automation.v8.auto_eq",
                        types.SimpleNamespace(auto_profile=fake, high_gap_db=fake_gap))
    lead = np.zeros(100, dtype=np.float32)
    raw_inst = np.zeros((2, 80), dtype=np.float32)
    src, cfg, info = engine.select_correction_eq("auto", None, lead, 44100, True, raw_lead=lead[:90],
                                                 raw_inst=raw_inst, voice="ryo")
    assert src == "auto" and not ceq.is_flat(cfg)
    assert seen == {"n": 100, "sr": 44100, "voice": "ryo", "gap": 2.345, "gap_args": (90, 1)}
    assert info == {"high_gap_db": 2.35, "broad_high": broad}
    src, cfg, info = engine.select_correction_eq("auto", None, lead, 44100, True)
    assert seen["gap"] is None and info["high_gap_db"] is None
    assert engine.select_correction_eq("auto", None, lead, 44100, False) == ("none", None, {})
    assert engine.select_correction_eq("voice", EQ, lead, 44100, True) == ("voice", EQ, {})


def _export_script():
    spec = importlib.util.spec_from_file_location("export_song_script", ROOT / "scripts" / "s1" / "export_song.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_export_song_uses_applied_eq(tmp_path):
    mod = _export_script()
    assert mod.correction_eq_path(tmp_path, None) is None
    applied = ceq.save(EQ, tmp_path / engine.APPLIED_CEQ_NAME)
    assert mod.correction_eq_path(tmp_path, None) == applied
    assert mod.correction_eq_path(tmp_path, tmp_path / "other.json") == tmp_path / "other.json"
