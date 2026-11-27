"""R3 (v8 fixed chain): knobs and the self-made side-chain DSP. No plugin is loaded here."""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest

from mix_automation.v8 import dsp
from mix_automation.v8.knobs import KNOB_SPEC_MAP, KNOB_SPECS, V8Knobs, load_knobs

SR = 44100
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ knobs


def test_knob_defaults_inside_range_and_match_dataclass():
    k = V8Knobs()
    assert {f.name for f in fields(k)} == {s.name for s in KNOB_SPECS}
    for s in KNOB_SPECS:
        assert s.lo <= s.default <= s.hi
        assert getattr(k, s.name) == pytest.approx(s.default)


def test_knob_out_of_range_raises():
    spec = KNOB_SPEC_MAP["deess_db"]
    with pytest.raises(ValueError):
        load_knobs(None, {"deess_db": spec.hi + 0.1})
    with pytest.raises(ValueError):
        load_knobs(None, {"no_such_knob": 1.0})


def test_knob_json_then_override(tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"brightness": 4.0, "glue_wet": 0.1}), encoding="utf-8")
    k = load_knobs(p, {"glue_wet": 0.5})
    assert k.brightness == 4.0
    assert k.glue_wet == 0.5
    assert k.saturation == V8Knobs().saturation


# ------------------------------------------------------------------ helpers


def _bandlimited_noise(n: int, lo: float, hi: float, ch: int = 1) -> np.ndarray:
    x = RNG.standard_normal((ch, n))
    y = dsp.butter_filter(x, SR, "bandpass", (lo, hi), order=4)
    return y if ch > 1 else y[0]


def _band_db(x: np.ndarray, lo: float, hi: float) -> float:
    y = dsp.butter_filter(x, SR, "bandpass", (lo, hi), order=4)
    return 10 * np.log10(np.mean(y**2) + 1e-20)


# ------------------------------------------------------------------ multiband duck


def test_multiband_duck_silent_key_is_transparent():
    n = SR * 2
    inst = 0.1 * RNG.standard_normal((2, n))
    out, st = dsp.multiband_duck(inst, np.zeros(n), SR, max_db=6.0)
    core = slice(4096, n - 4096)
    err = np.max(np.abs(out[:, core] - inst[:, core]))
    assert err < 1e-6
    assert all(b["max_gr_db"] == 0.0 for b in st["bands"])


def test_multiband_duck_only_target_band_goes_down():
    n = SR * 3
    inst = 0.05 * RNG.standard_normal((2, n))
    key = 0.5 * _bandlimited_noise(n, 1300.0, 2500.0)  # inside band 2 (1-3 kHz, weight 1.0)
    out, st = dsp.multiband_duck(inst, key, SR, max_db=6.0)
    core = slice(SR // 2, n - SR // 2)
    d_target = _band_db(inst[:, core], 1300, 2500) - _band_db(out[:, core], 1300, 2500)
    d_low = _band_db(inst[:, core], 60, 150) - _band_db(out[:, core], 60, 150)
    d_high = _band_db(inst[:, core], 9000, 14000) - _band_db(out[:, core], 9000, 14000)
    assert d_target > 4.0
    assert abs(d_low) < 0.1
    assert abs(d_high) < 0.1
    assert st["bands"][1]["max_gr_db"] == pytest.approx(6.0, abs=0.2)
    g_l = _band_db(out[0:1, core], 1300, 2500) - _band_db(inst[0:1, core], 1300, 2500)
    g_r = _band_db(out[1:2, core], 1300, 2500) - _band_db(inst[1:2, core], 1300, 2500)
    assert g_l == pytest.approx(g_r, abs=0.3)


# ------------------------------------------------------------------ M/S compressor, doubler post


def test_ms_compress_leaves_side_untouched():
    n = SR
    mid = 0.5 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)
    side = 0.1 * RNG.standard_normal(n)
    x = np.stack([mid + side, mid - side])
    y, st = dsp.ms_compress(x, SR, thr_db=-20.0, ratio=2.0)
    side_out = 0.5 * (y[0] - y[1])
    mid_out = 0.5 * (y[0] + y[1])
    assert np.max(np.abs(side_out - side)) < 1e-5
    assert np.sqrt(np.mean(mid_out**2)) < np.sqrt(np.mean(mid**2)) * 0.9
    assert st["mean_gr_db"] > 1.0


def test_doubler_post_mid_minus_6db():
    n = SR
    t = np.arange(n) / SR
    mid = 0.3 * np.sin(2 * np.pi * 300 * t)
    side = 0.2 * np.sin(2 * np.pi * 500 * t)
    y = dsp.doubler_post(np.stack([mid + side, mid - side]), SR)
    core = slice(SR // 10, n)
    m_out = 0.5 * (y[0] + y[1])[core]
    s_out = 0.5 * (y[0] - y[1])[core]
    ratio_mid = 20 * np.log10(np.sqrt(np.mean(m_out**2)) / np.sqrt(np.mean(mid[core] ** 2)))
    ratio_side = 20 * np.log10(np.sqrt(np.mean(s_out**2)) / np.sqrt(np.mean(side[core] ** 2)))
    assert ratio_mid == pytest.approx(-6.0, abs=0.15)
    assert ratio_side == pytest.approx(0.0, abs=0.15)


# ------------------------------------------------------------------ ping-pong delay


def test_pingpong_first_tap_left_second_right():
    n = SR
    x = np.zeros(n)
    x[0] = 1.0
    d = 0.1
    y = dsp.pingpong_delay(x, SR, d, feedback=0.5)
    first_l = np.argmax(np.abs(y[0, : int(1.5 * d * SR)])) / SR
    first_r = np.argmax(np.abs(y[1, : int(2.5 * d * SR)])) / SR
    assert first_l == pytest.approx(d, abs=0.003)
    assert first_r == pytest.approx(2 * d, abs=0.003)
    assert np.max(np.abs(y[1, : int(1.5 * d * SR)])) < 1e-9  # nothing on R before the 2nd tap


# ------------------------------------------------------------------ phrase ride


def test_phrase_ride_brings_loud_and_quiet_phrases_inside_tolerance():
    n = SR * 10
    lead = np.zeros(n)
    levels = [0.3, 0.15, 0.2, 0.4]  # 8.5 dB spread, inside the +-6 dB correction range
    spans = []
    for i, a in enumerate(levels):
        s0, s1 = int((0.5 + 2.4 * i) * SR), int((2.0 + 2.4 * i) * SR)
        lead[s0:s1] = a * _bandlimited_noise(s1 - s0, 200.0, 4000.0)
        spans.append((s0, s1))
    inst = 0.05 * RNG.standard_normal((2, n))
    target, tol = -2.5, 1.5
    res = dsp.phrase_ride(lead, inst, SR, target_lu=target, tol_lu=tol)
    assert len(res.phrases) == len(levels)
    k_out = dsp.k_weight(res.audio, SR)
    k_inst = dsp.k_weight(inst, SR)
    for s0, s1 in spans:
        a, b = s0 + int(0.1 * SR), s1 - int(0.1 * SR)
        d = dsp.lufs_ungated(2 * np.mean(k_out[a:b] ** 2)) - dsp.lufs_ungated(np.sum(np.mean(k_inst[:, a:b] ** 2, axis=1)))
        assert abs(d - target) <= tol + 0.3


# ------------------------------------------------------------------ calibrate, duck_gain


def test_calibrate_converges_on_monotonic_function():
    def render(p: float) -> float:
        return p

    res = dsp.calibrate(render, lambda y: 3.0 * y**1.3 + 1.0, target=10.0, lo=0.0, hi=10.0, iters=4, tol=0.05)
    assert res.converged
    assert 3.0 * res.value**1.3 + 1.0 == pytest.approx(10.0, abs=0.05)


def test_calibrate_decreasing_response_and_unreachable_target():
    res = dsp.calibrate(lambda p: p, lambda y: -2.0 * y, target=-5.0, lo=0.0, hi=10.0, tol=0.05)
    assert res.value == pytest.approx(2.5, abs=0.05)
    bad = dsp.calibrate(lambda p: p, lambda y: y, target=50.0, lo=0.0, hi=10.0)
    assert not bad.converged and bad.value == 10.0


def test_duck_gain_silent_key_is_unity():
    g = dsp.duck_gain(np.zeros(SR), SR, depth_db=6.0, thr_db=-40.0, range_db=15.0)
    assert np.allclose(g, 1.0)


def test_duck_gain_loud_key_reaches_depth():
    key = 0.5 * RNG.standard_normal(SR)
    g = dsp.duck_gain(key, SR, depth_db=6.0, thr_db=-40.0, range_db=15.0)
    assert 20 * np.log10(g[SR // 2]) == pytest.approx(-6.0, abs=0.01)


def test_delay_note_doubles_when_sixteenth_too_short():
    from mix_automation.v8.chain import delay_note_s

    d, n = delay_note_s(142.9)
    assert n == "1/16" and abs(d - 60 / 142.9 / 4) < 1e-9
    d, n = delay_note_s(187.5)
    assert n == "1/8" and abs(d - 0.16) < 1e-6
