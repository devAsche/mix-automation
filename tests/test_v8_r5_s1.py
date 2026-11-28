"""v8 R5 part B: Studio One export / read-back of the correction EQ, and average_profiles.

Plugin tests use the real VST3s (Persistent Q, Saturation Knob, VCA Compressor, Wave Breaker) and are
skipped when one is missing. The v8 folder is synthetic (4 s of noise / tone) but its master and
mix_report stages come from the real chain.master_bus.
"""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import pytest
import soundfile as sf

from mix_automation.v8 import correction_eq as ceq
from mix_automation.v8 import s1_preset as sp

SR = 44100


# ---------------------------------------------------------------- pure parts


def test_juce_base64_roundtrip():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 5, 63, 64, 65, 1000):
        d = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        assert sp.decode_juce_base64(sp.encode_juce_base64(d)) == d


def test_raw_state_with_streams_rewraps_length():
    xml = (b'<?xml version="1.0" encoding="UTF-8"?> <VST3PluginState><IComponent>'
           + sp.encode_juce_base64(b"old").encode() + b"</IComponent></VST3PluginState>")
    tmpl = b"VC2!" + len(xml).to_bytes(4, "little") + xml + b"\x00"
    out = sp.raw_state_with_streams(tmpl, b"new component", b"ctrl")
    n = int.from_bytes(out[4:8], "little")
    assert out[:4] == b"VC2!" and len(out) == 8 + n + 1 and out.endswith(b"\x00")
    from mix_automation.io.song_builder import extract_steinberg_comp_and_cont

    assert extract_steinberg_comp_and_cont(out) == (b"new component", b"ctrl")


def test_parse_vstpreset_roundtrip():
    from mix_automation.io.song_builder import create_vstpreset_bytes

    cid = "{ABCDEF01-9182-FAEB-4B61-706550713031}"
    b = create_vstpreset_bytes(cid, b"COMP", b"CONT")
    p = sp.parse_vstpreset(b)
    assert p["class_id"] == "ABCDEF019182FAEB4B61706550713031"
    assert p["chunks"]["Comp"] == b"COMP" and p["chunks"]["Cont"] == b"CONT"
    sp.check_class(b, cid)
    with pytest.raises(sp.S1PresetError):
        sp.check_class(b, "{24DCAB96-A33C-41E5-B2C9-7A9953B51CCD}")
    with pytest.raises(sp.S1PresetError):
        sp.parse_vstpreset(b"nope" * 20)


def test_zip_member_name_cp932():
    info = zipfile.ZipInfo("x")
    info.filename = "Presets/Channels/メイン/1.vstpreset".encode("cp932").decode("cp437")
    info.flag_bits = 0
    assert sp.zip_member_name(info) == "Presets/Channels/メイン/1.vstpreset"


def test_read_script_never_overwrites(tmp_path):
    import importlib.util
    from datetime import datetime

    spec = importlib.util.spec_from_file_location(
        "read_correction_eq", os.path.join(os.path.dirname(__file__), "..", "scripts", "s1", "read_correction_eq.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / "ryo.json"
    assert mod.free_path(p) == p
    p.write_text("{}")
    alt = mod.free_path(p, datetime(2026, 11, 28, 1, 2, 3))
    assert alt.name == "ryo_20261128-010203.json"


# ---------------------------------------------------------------- B4 average_profiles


def _cfg(bands):
    return ceq.validate({"format": ceq.FORMAT, "bands": bands})


def test_rbj_response_basics():
    f = ceq.log_grid()
    assert f[0] == pytest.approx(20.0) and f[-1] == pytest.approx(20000.0, rel=1e-6)
    pk = _cfg([{"band": 1, "type": "Peaking", "frequency_hz": 1000.0, "gain_db": -6.0, "q": 1.0}])
    r = ceq.response_db(pk, f)
    assert r[np.argmin(abs(f - 1000.0))] == pytest.approx(-6.0, abs=0.05)
    assert abs(r[0]) < 0.05 and abs(r[-1]) < 0.05
    hs = _cfg([{"band": 1, "type": "HighShelf", "frequency_hz": 5000.0, "gain_db": 3.0, "q": 0.707}])
    r = ceq.response_db(hs, f)
    assert r[-1] == pytest.approx(3.0, abs=0.1) and abs(r[0]) < 0.05
    assert ceq.response_db(None, f) == pytest.approx(np.zeros_like(f))


def test_average_profiles_recovers_mean():
    a = _cfg([{"band": 1, "type": "Peaking", "frequency_hz": 3000.0, "gain_db": -2.0, "q": 1.4}])
    b = _cfg([{"band": 4, "type": "Peaking", "frequency_hz": 3000.0, "gain_db": -4.0, "q": 1.4}])
    avg = ceq.average_profiles([a, b], n_bands=1)
    assert len(avg["bands"]) == 1
    bd = avg["bands"][0]
    assert bd["type"] == "Peaking"
    assert bd["frequency_hz"] == pytest.approx(3000.0, rel=0.05)
    assert bd["gain_db"] == pytest.approx(-3.0, abs=0.1)
    f = ceq.log_grid()
    target = 0.5 * (ceq.response_db(a, f) + ceq.response_db(b, f))
    assert np.max(np.abs(ceq.response_db(avg, f) - target)) < 0.1
    assert avg["source"]["averaged_from"] == 2


def test_average_profiles_mixed_shapes_fit():
    a = _cfg([{"band": 1, "type": "LowShelf", "frequency_hz": 200.0, "gain_db": -1.5, "q": 0.7},
              {"band": 2, "type": "Peaking", "frequency_hz": 3200.0, "gain_db": -2.0, "q": 1.4}])
    b = _cfg([{"band": 1, "type": "HighShelf", "frequency_hz": 10000.0, "gain_db": 2.0, "q": 0.7},
              {"band": 2, "type": "Peaking", "frequency_hz": 3000.0, "gain_db": -1.0, "q": 1.0}])
    c = _cfg([])
    avg = ceq.average_profiles([a, b, c], n_bands=4)
    f = ceq.log_grid()
    target = np.mean([ceq.response_db(x, f) for x in (a, b, c)], axis=0)
    err = ceq.response_db(avg, f) - target
    assert np.sqrt(np.mean(err ** 2)) < 0.1 and np.max(np.abs(err)) < 0.3
    assert ceq.validate(avg) == avg
    assert ceq.is_flat(ceq.average_profiles([c, c], n_bands=3))


def test_average_profiles_rejects():
    with pytest.raises(ceq.CorrectionEQError):
        ceq.average_profiles([])
    hp = _cfg([{"band": 1, "type": "Highpass", "frequency_hz": 100.0, "q": 0.7, "slope_db_oct": 18.0}])
    with pytest.raises(ceq.CorrectionEQError):
        ceq.average_profiles([hp])


# ---------------------------------------------------------------- plugin parts (skipped when absent)

NEEDED = ("persistent_q", "saturation_knob", "vca_compressor", "wave_breaker")


def _plugins_or_skip():
    from mix_automation.config.plugin_registry import get_plugin_map

    pm = get_plugin_map()
    for k in NEEDED:
        if k not in pm or not pm[k].loadable_path.exists():
            pytest.skip(f"{k} not installed")


CORR = {"format": ceq.FORMAT, "voice": "ryo", "bands": [
    {"band": 1, "type": "LowShelf", "frequency_hz": 200.0, "gain_db": -1.5, "q": 0.7},
    {"band": 2, "type": "Peaking", "frequency_hz": 3200.0, "gain_db": -2.0, "q": 1.4},
    {"band": 3, "type": "HighShelf", "frequency_hz": 10000.0, "gain_db": 1.0, "q": 0.7},
]}


@pytest.fixture(scope="module")
def v8_dir(tmp_path_factory):
    """Synthetic v8 output folder; master + stages from the real chain.master_bus."""
    _plugins_or_skip()
    from mix_automation.v8 import chain
    from mix_automation.v8.knobs import load_knobs

    d = tmp_path_factory.mktemp("v8") / "Test_Song_v8"
    (d / "stems").mkdir(parents=True)
    rng = np.random.default_rng(5)
    n = SR * 4
    t = np.arange(n) / SR
    lead = (0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(n))
    lead = lead.astype(np.float32)
    sends = {k: (0.03 * rng.standard_normal((2, n))).astype(np.float32) for k in ("doubler", "room", "plate", "delay")}
    inst = (0.1 * rng.standard_normal((2, n))).astype(np.float32)
    knobs = load_knobs()
    log = chain.StageLog()
    mix = inst.astype(np.float64) + chain.vocal_bus(lead, sends, n)
    master, _ = chain.master_bus(mix.astype(np.float32), SR, knobs, log)
    sf.write(str(d / "stems" / "lead_dry.wav"), lead, SR, subtype="FLOAT")
    for k, w in sends.items():
        sf.write(str(d / "stems" / f"wet_{k}.wav"), w.T, SR, subtype="FLOAT")
    sf.write(str(d / "stems" / "inst_ducked.wav"), inst.T, SR, subtype="FLOAT")
    sf.write(str(d / "Test_Song_v8_Master.wav"), master.T, SR, subtype="PCM_24")
    rep = {"song": "Test Song", "sr": SR, "tempo_bpm": 128.0, "knobs": knobs.to_dict(), "stages": log.stages}
    (d / "mix_report.json").write_text(json.dumps(rep, default=float), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def exported(v8_dir):
    from mix_automation.v8.s1_export import export_v8_song

    return export_v8_song(v8_dir, verify=True)


def test_export_files_and_check(exported, v8_dir):
    from mix_automation.v8 import s1_export as se

    out = v8_dir / "s1_export"
    assert exported["out_dir"] == str(out)
    assert (out / "Test_Song_v8r5.song").exists() and (out / "Test_Song_v8r5_orig.song").exists()
    assert (out / "Test_Song_v8r5.song").read_bytes() == (out / "Test_Song_v8r5_orig.song").read_bytes()
    assert exported["check_issues"] == []
    assert json.loads((out / "export_report.json").read_text(encoding="utf-8"))["check_issues"] == []
    m = sp.read_song_members(out / "Test_Song_v8r5.song")
    ins = sp.channel_inserts(sp.parse_s1_xml(m["Devices/audiomixer.xml"]))
    assert [s["plugin"] for s in ins[se.TRACK_CORR]] == ["Persistent Q"]
    assert [s["plugin"] for s in ins[se.MASTER_LABEL]] == ["Saturation Knob", "VCA Compressor", "Wave Breaker"]
    assert set(ins) >= {se.TRACK_INST, se.TRACK_REF, "Wet doubler", "Wet delay"}
    assert not any(k.startswith("Presets/Channels/mainvocal") for k in m)
    mixer = m["Devices/audiomixer.xml"].decode("utf-8")
    assert f'label="{se.TRACK_REF}" mute="1"' in mixer
    # a second export never writes into the first folder
    rep2 = se.export_v8_song(v8_dir, verify=False)
    assert rep2["out_dir"] != str(out)


def test_ref_track_is_sum_and_gain(exported, v8_dir):
    from mix_automation.v8 import s1_export as se

    r = se.read_v8_render(v8_dir)
    tr = {k: sf.read(p, dtype="float32", always_2d=True)[0].T for k, p in exported["tracks"].items()}
    others = sum(v.astype(np.float64) for k, v in tr.items() if k != se.TRACK_REF)
    assert np.max(np.abs(others - tr[se.TRACK_REF])) < 1e-5
    lead = sf.read(r.stems["lead_dry"], dtype="float32")[0]
    g = 10 ** (r.work_gain_db / 20)
    assert np.allclose(tr[se.TRACK_CORR][0], lead * g, atol=1e-6)


def test_master_reproduction(exported):
    v = exported["verify"]
    # plugin states from the written presets + the same M/S compressor: the v8 master nulls
    assert abs(v["with_ms_compressor"]["lufs_diff"]) < 0.05
    assert v["with_ms_compressor"]["null_db"] < -40.0
    # the .song chain (no M/S comp) lands within 0.1 LU after the drive re-set
    assert abs(v["song_chain"]["lufs_diff"]) <= 0.1


def test_check_song_catches_problems(exported, tmp_path):
    from mix_automation.v8 import s1_export as se

    src = exported["song"]
    m = sp.read_song_members(src)
    bad = tmp_path / "bad.song"
    with zipfile.ZipFile(bad, "w") as z:
        for k, v in m.items():
            if k == "Song/mediapool.xml":
                v = v.replace(b"s1_media/", b"s1_missing/")
            if k.endswith("Persistent Q.vstpreset"):
                continue
            z.writestr(k, v)
    issues = se.check_song(bad, load_presets=False)
    assert any("media URL" in i for i in issues)
    assert any("Persistent Q" in i and "missing" in i for i in issues)


# ---------------------------------------------------------------- B3 round trip


def test_correction_eq_round_trip_through_song(v8_dir, tmp_path):
    from mix_automation.v8 import chain
    from mix_automation.v8 import s1_export as se

    cfg = ceq.validate(CORR)
    rep = se.export_v8_song(v8_dir, tmp_path / "rt", correction_cfg=cfg, verify=False)
    assert rep["check_issues"] == []
    back = sp.correction_eq_from_song(rep["song"], voice="ryo")
    assert back["source"]["track"] == se.TRACK_CORR and len(back["source"]["song_sha256"]) == 64
    assert back["global"] == {}
    assert [b["band"] for b in back["bands"]] == [1, 2, 3]
    for a, b in zip(cfg["bands"], back["bands"], strict=True):
        assert a["type"] == b["type"] and a["placement"] == b["placement"]
        assert b["frequency_hz"] == pytest.approx(a["frequency_hz"], rel=0.01)
        assert b["q"] == pytest.approx(a["q"], rel=0.01)
        assert b["gain_db"] == pytest.approx(a["gain_db"], abs=0.05)  # plugin gain steps (as part A)
    rng = np.random.default_rng(9)
    x = (0.1 * rng.standard_normal(SR * 2)).astype(np.float32)
    y_a = chain.correction_eq(x, SR, cfg, chain.StageLog())
    y_b = chain.correction_eq(x, SR, back, chain.StageLog())
    res = 10 * np.log10(np.sum((y_a - y_b) ** 2) / np.sum(y_a ** 2) + 1e-30)
    print(f"round-trip bands {[(b['frequency_hz'], b['gain_db'], b['q']) for b in back['bands']]} null {res:.1f} dB")
    assert res < -60.0, res
