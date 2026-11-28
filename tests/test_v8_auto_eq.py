"""R5b: automatic correction EQ (v8.auto_eq)."""

from __future__ import annotations

import numpy as np
from scipy import signal

from mix_automation.v8 import auto_eq
from mix_automation.v8 import correction_eq as ceq

SR = 44100


def _voice_with_peaks(hz: list[float], seconds: float = 8.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(SR * seconds)).astype(np.float64)
    x = signal.sosfilt(signal.butter(1, 1500, "low", fs=SR, output="sos"), x)  # voice-like tilt
    for f in hz:  # narrow resonances, +8 dB
        b, a = signal.iirpeak(f, 20, fs=SR)
        x = x + 1.5 * signal.lfilter(b, a, x)
    env = (np.sin(2 * np.pi * 0.5 * np.arange(len(x)) / SR) > -0.3).astype(float)  # phrases and gaps
    return (0.1 * x * env).astype(np.float32)


def test_peaks_found_in_ranges_and_cut_only():
    cfg = auto_eq.auto_profile(_voice_with_peaks([3200.0, 11000.0]), SR)
    hz = [b["frequency_hz"] for b in cfg["bands"]]
    assert any(abs(np.log2(h / 3200.0)) < 1 / 6 for h in hz)
    assert any(abs(np.log2(h / 11000.0)) < 1 / 6 for h in hz)
    assert all(b["gain_db"] < 0 for b in cfg["bands"])
    counts = {}
    for p in cfg["source"]["peaks"]:
        counts[str(p["range"])] = counts.get(str(p["range"]), 0) + 1
    for lo, hi, n_max, *_ in auto_eq.RANGES:
        assert counts.get(str([lo, hi]), 0) <= n_max
    ceq.validate(cfg)


def test_broad_high_band_rule_points_and_limits():
    eva = auto_eq.broad_high_band(5.10)  # Ryo: net ~0 -> about Ryo's -2.55 dB band
    assert -2.6 < eva["gain_db"] < -2.0 and abs(eva["net_db"]) < 0.05
    ld = auto_eq.broad_high_band(8.07)  # net +1.5
    assert -0.6 < ld["gain_db"] < 0.0
    assert auto_eq.broad_high_band(20.0)["gain_db"] == 0.0  # never adds
    low = auto_eq.broad_high_band(-10.0)  # net target clipped at 0 -> same band as エバ, within -3..0
    assert low["net_target_db"] == 0.0 and low["gain_db"] == eva["gain_db"] and low["gain_db"] >= -3.0


def test_auto_profile_adds_broad_band_only_when_cutting():
    x = _voice_with_peaks([3200.0])
    with_band = auto_eq.auto_profile(x, SR, gap_db=5.1)
    without = auto_eq.auto_profile(x, SR, gap_db=9.0)
    assert any(b["frequency_hz"] == auto_eq.BROAD_HIGH_HZ for b in with_band["bands"])
    assert not any(b["frequency_hz"] == auto_eq.BROAD_HIGH_HZ for b in without["bands"])
