"""R5: fixed tone + relation-only distance (A1) and the correction EQ stage (A2)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mix_automation.v8 import chain
from mix_automation.v8 import correction_eq as ceq
from mix_automation.v8.knobs import (
    KNOB_SPEC_MAP,
    R5_TONE,
    V8Knobs,
    has_r5_tone,
    validate_knobs,
    with_r5_tone,
)
from mix_automation.v8.match import (
    STAGE_A,
    STAGE_A_R5,
    STAGE_B,
    TERMS,
    WEIGHTS_R4B,
    WEIGHTS_R5,
    SongMatch,
    distance,
)

SR = 44100

# ---------------------------------------------------------------- A1


def test_r5_tone_in_range_and_not_searched():
    k = with_r5_tone(V8Knobs())
    validate_knobs(k)
    assert has_r5_tone(k) and not has_r5_tone(V8Knobs())
    assert V8Knobs().hpf_hz == KNOB_SPEC_MAP["hpf_hz"].default  # engine default unchanged
    assert not set(R5_TONE) & set(STAGE_A_R5)
    assert not set(R5_TONE) & set(STAGE_B)
    assert set(STAGE_A_R5) <= set(STAGE_A)


def test_weights_r5_drop_spec_and_afx():
    assert WEIGHTS_R5["spec"] == 0.0 and WEIGHTS_R5["afx"] == 0.0
    assert {k: v for k, v in WEIGHTS_R5.items() if k not in ("spec", "afx")} == \
        {k: v for k, v in WEIGHTS_R4B.items() if k not in ("spec", "afx")}
    fa = {"lu": np.array([3.0]), "spec": np.zeros(3), "sm": -10.0}
    fb = {"lu": np.array([3.0]), "spec": np.full(3, 5.0), "sm": -10.0}
    scales = dict.fromkeys(TERMS, 1.0)
    d5, _ = distance(fa, fb, scales, WEIGHTS_R5, TERMS)
    d4, _ = distance(fa, fb, scales, WEIGHTS_R4B, TERMS)
    assert d5 == 0.0 and d4 > 0.0


def test_songmatch_r5_paths(tmp_path):
    sm = SongMatch("エバ", r5=True, cache_root=tmp_path)
    assert sm.r4b and sm.r5 and not sm.afx
    assert sm.sfx == "_r5" and sm.dir == tmp_path / "エバ"
    assert sm.weights == WEIGHTS_R5
    sm4 = SongMatch("エバ", r4b=True, cache_root=tmp_path)
    assert sm4.afx and sm4.sfx == "_r4b" and sm4.weights == WEIGHTS_R4B


# ---------------------------------------------------------------- A2: JSON


def _cfg(**band):
    b = {"band": 1, "enabled": True, "type": "Peaking", "frequency_hz": 3000.0, "gain_db": -2.0, "q": 1.0,
         "slope_db_oct": None, "placement": "Stereo", "extra": {}}
    b.update(band)
    return {"format": ceq.FORMAT, "voice": "ryo", "plugin": "Persistent Q", "global": {}, "bands": [b]}


def test_validate_and_roundtrip_file(tmp_path):
    cfg = ceq.validate(_cfg())
    p = ceq.save(cfg, tmp_path / "ryo.json")
    assert ceq.load(p) == cfg
    with pytest.raises(FileExistsError):
        ceq.save(cfg, p)
    assert ceq.load(tmp_path / "missing.json") is None
    assert ceq.resolve_path("ryo") == ceq.PROFILE_DIR / "ryo.json"
    assert ceq.resolve_path("ryo", tmp_path / "x.json") == tmp_path / "x.json"


@pytest.mark.parametrize("bad", [
    {"type": "Shelf"},
    {"frequency_hz": 10.0},
    {"gain_db": 31.0},
    {"q": 0.0},
    {"slope_db_oct": 9.0},
    {"placement": "Center"},
    {"band": 33},
    {"enabled": 1},
    {"extra": {"foo": 1}},
])
def test_validate_rejects(bad):
    with pytest.raises(ceq.CorrectionEQError):
        ceq.validate(_cfg(**bad))


def test_validate_rejects_top_level():
    with pytest.raises(ceq.CorrectionEQError):
        ceq.validate({**_cfg(), "format": "x"})
    with pytest.raises(ceq.CorrectionEQError):
        ceq.validate({**_cfg(), "global": {"output_gain": 30.0}})
    with pytest.raises(ceq.CorrectionEQError):
        ceq.validate({**_cfg(), "global": {"latency_mode": "Low"}})
    with pytest.raises(ceq.CorrectionEQError):
        ceq.validate({**_cfg(), "bands": [_cfg()["bands"][0]] * 2})


def test_is_flat():
    assert ceq.is_flat(None)
    assert not ceq.is_flat(ceq.validate(_cfg()))
    assert ceq.is_flat(ceq.validate(_cfg(enabled=False)))
    assert ceq.is_flat(ceq.validate(_cfg(gain_db=0.0)))
    assert ceq.is_flat(ceq.validate(_cfg(gain_db=0.0, type="HighShelf")))
    assert not ceq.is_flat(ceq.validate(_cfg(gain_db=0.0, type="Highpass", slope_db_oct=12.0)))
    assert not ceq.is_flat(ceq.validate(_cfg(gain_db=0.0, extra={"dynamic": True})))
    assert not ceq.is_flat(ceq.validate({**_cfg(gain_db=0.0), "global": {"output_gain": -1.0}}))
    assert ceq.is_flat(ceq.validate({**_cfg(), "global": {"bypass": True}}))
    assert ceq.is_flat(ceq.validate({**_cfg(), "bands": []}))


# ---------------------------------------------------------------- A2: chain, flat = bit identical


def test_flat_correction_is_bit_identical_and_loads_nothing(monkeypatch, tmp_path):
    def no_load(*_a, **_k):
        raise AssertionError("plugin must not be loaded for a flat correction EQ")

    monkeypatch.setattr(chain, "load_stage_plugin", no_load)
    rng = np.random.default_rng(1)
    n = 4410
    lead = rng.standard_normal(n).astype(np.float32) * 0.1
    sends = {"room": rng.standard_normal((2, n)).astype(np.float32) * 0.01,
             "plate": rng.standard_normal((2, n - 10)).astype(np.float32) * 0.01}
    ref = chain.vocal_bus(lead, sends, n)
    flat = ceq.save(_cfg(gain_db=0.0), tmp_path / "flat.json")
    for cfg in (ceq.load(tmp_path / "missing.json"), ceq.load(flat), ceq.validate(_cfg(enabled=False))):
        log = chain.StageLog()
        out = chain.correction_eq(lead, SR, cfg, log)
        assert out is lead and not log.stages
        assert np.array_equal(chain.vocal_bus(out, sends, n), ref)


# ---------------------------------------------------------------- A2: real Persistent Q (skipped when absent)


def _pq_or_skip():
    from mix_automation.config.plugin_registry import get_plugin_map

    spec = get_plugin_map().get("persistent_q")
    if spec is None or not spec.loadable_path.exists():
        pytest.skip("Persistent Q not installed")
    from pedalboard import load_plugin

    return lambda: load_plugin(str(spec.loadable_path))


def _null_db(x, y):
    return 10 * np.log10(np.sum((y - x) ** 2) / np.sum(x ** 2) + 1e-30)


def test_persistent_q_all_bands_off_nulls_and_no_latency():
    load = _pq_or_skip()
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(SR) * 0.1).astype(np.float32)
    x[1000] = 0.9  # impulse to see any delay
    plug = load()
    ceq.write_to_plugin(plug, ceq.validate({**_cfg(), "bands": []}))
    y = chain.run_plugin(plug, x, SR)
    assert _null_db(x, y) < -90.0
    # one band on: still aligned (Persistent Q zero-latency mode; the linear-phase latency of 1024
    # samples is compensated by pedalboard, measured 2026-11-28)
    log = chain.StageLog()
    y = chain.correction_eq(x, SR, ceq.validate(_cfg(gain_db=-3.0)), log)
    assert y is not x and log.stages[0]["stage"] == "lead_correction_eq"
    assert log.stages[0]["key"] == chain.PERSISTENT_Q_CORRECTION_KEY
    c = np.correlate(y[:8000], x[:8000], "full")
    assert int(np.argmax(np.abs(c))) - (8000 - 1) == 0


def test_persistent_q_apply_then_read_back():
    load = _pq_or_skip()
    cfg = ceq.validate({
        "format": ceq.FORMAT, "voice": "ryo", "plugin": "Persistent Q", "global": {"output_gain": -1.5},
        "bands": [
            {"band": 1, "type": "Highpass", "frequency_hz": 120.0, "q": 0.7, "slope_db_oct": 18.0},
            {"band": 3, "type": "Peaking", "frequency_hz": 3000.0, "gain_db": -2.0, "q": 1.4, "placement": "Mid"},
            {"band": 7, "type": "HighShelf", "frequency_hz": 10000.0, "gain_db": 1.5, "q": 0.7,
             "extra": {"dynamic": True, "threshold": -30.0}},
        ]})
    ref = ceq.fresh_reference()
    plug = load()
    ceq.write_to_plugin(plug, cfg)
    back = ceq.bands_from_plugin(plug, ref, voice="ryo")
    assert back["global"].keys() == {"output_gain"}
    assert back["global"]["output_gain"] == pytest.approx(-1.5, abs=0.05)
    assert [b["band"] for b in back["bands"]] == [1, 3, 7]
    for a, b in zip(cfg["bands"], back["bands"], strict=True):
        assert (a["type"], a["placement"]) == (b["type"], b["placement"])
        for k in ("frequency_hz", "q"):
            assert b[k] == pytest.approx(a[k], rel=0.02)
        assert b["gain_db"] == pytest.approx(a["gain_db"], abs=0.05)
    assert back["bands"][0]["slope_db_oct"] == 18.0 and back["bands"][1]["slope_db_oct"] is None
    assert back["bands"][2]["extra"]["dynamic"] is True
    assert back["bands"][2]["extra"]["threshold"] == pytest.approx(-30.0, abs=0.1)
    # and the JSON written from the plugin is valid and loadable
    json.dumps(back)
    assert ceq.validate(back) == back
