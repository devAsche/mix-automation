import numpy as np

from mix_automation import debug_taps
from mix_automation.dsp.vst3_optimizer import process_choir_bus, process_main_harmony_bus


def _signals(n: int = 44100 * 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(n) / 44100
    lead = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(n)).astype(np.float32)
    harm = np.stack([lead * 0.5, np.roll(lead, 37) * 0.5]).astype(np.float32)
    return lead, harm


def test_taps_record_nothing_outside_capture():
    debug_taps.drain()
    debug_taps.tap("x", np.ones(4))
    assert not debug_taps.active()
    assert debug_taps.drain() == {}
    with debug_taps.capture("p_", enabled=False):
        debug_taps.tap("x", np.ones(4))
    assert debug_taps.drain() == {}


def test_taps_copy_and_prefix():
    a = np.zeros(4, dtype=np.float32)
    with debug_taps.capture("p_"):
        assert debug_taps.active()
        debug_taps.tap("x", a)
    a[0] = 1.0
    got = debug_taps.drain()
    assert list(got) == ["p_x"] and got["p_x"][0] == 0.0
    assert debug_taps.drain() == {}


def test_harmony_bus_output_unchanged_by_taps():
    lead, harm = _signals()
    for fn in (process_main_harmony_bus, process_choir_bus):
        ref = fn(harm, lead, sr=44100)
        with debug_taps.capture("h_"):
            out = fn(harm, lead, sr=44100)
        taps = debug_taps.drain()
        np.testing.assert_array_equal(ref, out)
        assert {"h_20_in", "h_21_eq", "h_22_comp", "h_23_sib", "h_24_space", "h_25_level"} <= set(taps)
