"""R4: knob ranges, new DSP, main-only regions, distance terms, LOO selection. No plugin is loaded."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mix_automation.v8 import dsp
from mix_automation.v8.chain import render_delay
from mix_automation.v8.knobs import (
    KNOB_SPEC_MAP,
    R3_DEFAULTS,
    V8Knobs,
    load_knobs,
    pultec_air_kcs,
    r3_knobs,
)
from mix_automation.v8.match import (
    STAGE_A,
    STAGE_B,
    TERMS,
    Layout,
    Region,
    candidate_regions,
    distance,
    fallback_scales,
    features,
    from_unit,
    loo_select,
    orig_to_cover,
    select_regions,
    term_raw,
    to_unit,
)

SR = 44100
RNG = np.random.default_rng(11)


# ------------------------------------------------------------------ knobs


def test_r4_ranges_and_defaults():
    expect = {"vocal_offset_lu": (-3.0, 5.0, 1.0), "inst_duck_max_db": (0.0, 3.0, 0.5),
              "comp_total_gr_db": (3.0, 11.0, 6.0), "hpf_hz": (70.0, 180.0, 100.0),
              "presence_hz": (2000.0, 5000.0, 3000.0), "presence_db": (0.0, 3.0, 0.0),
              "air_hz": (8000.0, 12000.0, 8000.0), "delay_type": (0.0, 2.0, 0.0),
              "inst_side_high_db": (0.0, 2.0, 0.0), "par_comp_mix": (0.0, 0.3, 0.0),
              "pre_level_strength": (0.0, 1.0, 0.0)}
    for name, (lo, hi, d) in expect.items():
        s = KNOB_SPEC_MAP[name]
        assert (s.lo, s.hi, s.default) == (lo, hi, d)


def test_r3_defaults_inside_ranges():
    for name, v in R3_DEFAULTS.items():
        s = KNOB_SPEC_MAP[name]
        if name == "delay_type":
            assert v == -1.0  # R3 ping-pong, accepted outside the searched range
            continue
        assert s.lo <= v <= s.hi, name
    k = r3_knobs()
    assert (k.vocal_offset_lu, k.comp_total_gr_db, k.inst_duck_max_db, k.hpf_hz) == (-2.5, 8.5, 3.0, 140.0)


def test_old_r3_json_fills_new_knobs(tmp_path):
    old = {"vocal_offset_lu": -2.5, "ride_tolerance_lu": 1.5, "comp_total_gr_db": 8.5, "brightness": 2.5,
           "deess_db": 3.0, "saturation": 1.5, "doubler_send_db": -18.5, "room_send_db": -16.0,
           "plate_send_db": -18.0, "plate_decay_s": 1.35, "delay_send_db": -24.0, "fx_duck_db": 6.0,
           "inst_duck_max_db": 3.0, "glue_wet": 0.35}
    p = tmp_path / "r3.json"
    p.write_text(json.dumps(old), encoding="utf-8")
    k = load_knobs(p)
    assert k.comp_total_gr_db == 8.5 and k.hpf_hz == V8Knobs().hpf_hz and k.delay_type == 0.0


@pytest.mark.parametrize("bad", [0.5, 3.0, -2.0, 1.2])
def test_delay_type_validate(bad):
    with pytest.raises(ValueError):
        load_knobs(None, {"delay_type": bad})


@pytest.mark.parametrize("ok", [-1.0, 0.0, 1.0, 2.0])
def test_delay_type_valid(ok):
    assert load_knobs(None, {"delay_type": ok}).delay_type == ok


@pytest.mark.parametrize(("hz", "kcs"), [(8000, "8 KCS"), (8999, "8 KCS"), (9001, "10 KCS"), (11000, "10 KCS"),
                                         (11001, "12 KCS"), (12000, "12 KCS")])
def test_air_rounding(hz, kcs):
    s, used = pultec_air_kcs(hz)
    assert s == kcs and used == float(kcs.split()[0]) * 1000


def test_unit_roundtrip():
    k = replace_knobs(hpf_hz=123.0, presence_db=1.7, vocal_offset_lu=2.2, inst_side_high_db=0.4)
    for names in (STAGE_A, STAGE_B):
        u = to_unit(k, names)
        assert np.all((u >= 0) & (u <= 1))
        k2 = from_unit(u, names, V8Knobs())
        for n in names:
            assert getattr(k2, n) == pytest.approx(getattr(k, n))


def replace_knobs(**kw) -> V8Knobs:
    return load_knobs(None, kw)


# ------------------------------------------------------------------ new DSP


def test_feedback_delay_single_tap_centred():
    x = np.zeros(SR)
    x[100] = 1.0
    y = dsp.feedback_delay(x, SR, 0.1, feedback=0.0, hp=20.0, lp=20000.0, n_taps=1)
    assert y.shape == (2, SR) and np.allclose(y[0], y[1])
    assert abs(int(np.argmax(np.abs(y[0]))) - (100 + 4410)) < 10


def test_render_delay_types():
    x = RNG.standard_normal(SR).astype(np.float32) * 0.1
    for t in (-1, 0, 1, 2):
        y, rec = render_delay(x, SR, t, 120.0)
        assert y.shape == (2, SR) and rec["delay_type"] == t
    assert render_delay(x, SR, 2, 120.0)[1]["delay_s"] == 0.12
    with pytest.raises(ValueError):
        render_delay(x, SR, 3, 120.0)


def test_side_high_shelf_leaves_mid():
    x = RNG.standard_normal((2, SR)) * 0.1
    y = dsp.side_high_shelf(x, SR, 2.0, 6000.0)
    assert np.allclose(0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), atol=1e-5)
    hi = dsp.butter_filter(0.5 * (y[0] - y[1]), SR, "highpass", 10000.0, order=4)
    hx = dsp.butter_filter(0.5 * (x[0] - x[1]), SR, "highpass", 10000.0, order=4)
    assert 10 * np.log10(np.mean(hi ** 2) / np.mean(hx ** 2)) == pytest.approx(2.0, abs=0.3)


def _phrased(levels_db: list[float], on_s: float = 1.0, off_s: float = 0.6) -> np.ndarray:
    parts = []
    for lv in levels_db:
        parts.append(RNG.standard_normal(int(on_s * SR)) * 10 ** (lv / 20) * 0.1)
        parts.append(np.zeros(int(off_s * SR)))
    return np.concatenate(parts)


def test_pre_level_narrows_phrase_spread():
    x = _phrased([0, -8, 0, 4, -4])
    y, st = dsp.pre_level(x, SR, 1.0)
    assert st["n_phrases"] >= 4
    seg = int(1.6 * SR)

    def spread(z):
        lv = [10 * np.log10(np.mean(z[i * seg + 2205:i * seg + SR - 2205] ** 2)) for i in range(5)]
        return max(lv) - min(lv)

    assert spread(y) < spread(x) - 4.0
    assert np.allclose(dsp.pre_level(x, SR, 0.0)[0], x, atol=1e-6)


def test_parallel_compress_raises_quiet_parts():
    x = _phrased([0, -12, 0, -12])
    y, st = dsp.parallel_compress(x, SR, 0.3)
    assert st["wet_gr_db"] > 5.0
    seg = int(1.6 * SR)
    loud = np.mean(y[: SR] ** 2) / np.mean(x[: SR] ** 2)
    quiet = np.mean(y[seg: seg + SR] ** 2) / np.mean(x[seg: seg + SR] ** 2)
    assert quiet > loud


# ------------------------------------------------------------------ time mapping and regions


def test_orig_to_cover_constant_and_linear():
    const = np.array([[0.0, 441.0], [1e4, 441.0]])
    assert float(orig_to_cover(1000.0, const)) == pytest.approx(1441.0)
    lin = np.array([[0.0, 0.0], [1e4, 1e4 * SR * 1e-5]])  # 10 ppm drift
    o = 100.0 * SR
    c = float(orig_to_cover(o, lin))
    assert c - np.interp(c / SR, lin[:, 0], lin[:, 1]) == pytest.approx(o, abs=1e-3)


def test_candidate_regions_main_only():
    dt = 0.01
    n = 3200  # 32 s (the 2 s post-padding of the last region must fit)
    lead = np.zeros(n, bool)
    lead[200:900] = True   # 2..9 s singing
    lead[1200:2900] = True  # 12..29 s singing
    non = np.zeros(n, bool)
    non[1500:2000] = True  # harmonies 15..20 s
    regs = candidate_regions(lead, non, n * dt, frame_dt=dt)
    spans = [(r.t0, r.t1) for r in regs]
    assert any(abs(a - 2.0) < 0.02 and abs(b - 9.0) < 0.02 for a, b in spans)
    for a, b in spans:  # nothing overlaps the harmony part, all within length limits
        assert b <= 15.0 or a >= 20.0
        assert 2.0 <= b - a <= 10.0
    # 20..29 s (9 s) stays one region; 12..15 s (3 s) too
    assert any(abs(a - 20.0) < 0.02 for a, _ in spans) and any(abs(a - 12.0) < 0.02 for a, _ in spans)


def test_select_regions_low_sm_and_budget():
    cands = [Region(2.0, 9.0), Region(12.0, 15.0), Region(20.0, 29.0), Region(40.0, 45.0)]
    ref = {"A": np.linspace(-20, -5, 31), "B": np.linspace(-18, -6, 31)}
    sm = {"A": [-18.0, -6.0, -15.0, -17.0], "B": [-17.0, -7.0, -16.0, -12.0]}
    chosen = select_regions(cands, sm, ref, budget_s=30.0)
    ts = [d["region"].t0 for d in chosen]
    assert 12.0 not in ts  # high S/M for both covers
    assert sum(d["region"].padded_s for d in chosen) <= 30.0
    assert ts == sorted(ts)


# ------------------------------------------------------------------ features / distance


def _scene(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, Layout]:
    rng = np.random.default_rng(seed)
    lay = Layout((Region(1.0, 4.0), Region(6.0, 9.5)))
    n = lay.n
    t = np.arange(n) / SR
    gate = (np.sin(2 * np.pi * 0.55 * t) > -0.2).astype(float)  # ~1.2 s phrases, ~0.6 s gaps
    v = dsp.butter_filter(rng.standard_normal(n), SR, "bandpass", (150.0, 6000.0), order=2) * gate * 0.1
    voice = np.stack([v, v]) + 0.01 * rng.standard_normal((2, n)) * gate
    raw = 0.05 * rng.standard_normal((2, n))
    inst = raw * 0.9
    return voice, inst, raw, lay


def test_distance_zero_for_identical_and_grows():
    voice, inst, raw, lay = _scene()
    f0 = features(voice, inst, raw, lay)
    sc = fallback_scales()
    d0, norm0 = distance(f0, f0, sc)
    assert d0 == pytest.approx(0.0, abs=1e-12)
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in norm0.values() if np.isfinite(v))

    louder = features(voice * 10 ** (2 / 20), inst, raw, lay)
    assert term_raw(louder, f0)["lu"] == pytest.approx(4.0, abs=0.2)
    assert distance(louder, f0, sc)[0] > 0

    side = voice.copy()
    side[0] += 0.03 * voice[0]
    side[1] -= 0.03 * voice[1]
    assert term_raw(features(side, inst, raw, lay), f0)["sm"] > 0

    dark = np.stack([dsp.butter_filter(c, SR, "lowpass", 3000.0) for c in inst])
    t_dark = term_raw(features(voice, dark, raw, lay), f0)
    assert t_dark["inst_spec"] > 0 and t_dark["spec"] > 0

    assert set(norm0) == set(TERMS)


def test_loo_select_picks_by_other_covers():
    recs = [
        {"per_cover": {"a": 1.0, "b": 1.0, "c": 5.0}},
        {"per_cover": {"a": 0.2, "b": 3.0, "c": 3.0}},
        {"per_cover": {"a": 2.0, "b": 2.0, "c": 0.1}},
    ]
    out = loo_select(recs, ["a", "b", "c"])
    assert out["c"]["index"] == 0 and out["c"]["held_out"] == 5.0  # median(a,b) smallest at 0
    assert out["a"]["index"] == 2 and out["a"]["held_out"] == 2.0  # median(b,c): 3.0, 3.0, 1.05
    assert out["b"]["index"] == 2  # median(a,c): 3.0, 1.6, 1.05
