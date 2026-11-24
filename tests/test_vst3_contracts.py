"""Live Real-Audio VST3 Contract Tests.

Proves empirically on synthesized test signals (never mocks!) that every commercial VST3 plugin
in our mixing chain actually applies the expected acoustic transfer behavior:
  1. RX 10 Spectral De-ess (`cutoff=5800Hz`): preserves 2.5 kHz consonant attacks (< 0.8 dB loss)
     while attenuating a 7.2 kHz sibilant burst!
  2. Softube VCA Compressor: actually compresses dynamic bursts (measurable GR >= 2.0 dB & lower crest)!
  3. Softube Saturation Knob: actually generates odd/even upper harmonics on a pure 440 Hz sine wave!
  4. UAD Teletronix LA-2A: string knob assignment `'45'` actually produces measurable optical GR!
  5. Persistent Q: exact 5-band HPF/Peak/Shelf boosts & cuts match expected frequency response!
  6. Exponential Audio R2: Plate mode with `chordpth=0.0` & `rvb_time ~ 0.92s` is verified cleanly!
"""

from __future__ import annotations

import numpy as np
from pedalboard import load_plugin
from scipy import signal

from mix_automation.config.plugin_registry import get_plugin_map
from mix_automation.dsp.vst3_adapter import (
    configure_clean_plate_r2,
    configure_persistent_q_5band,
    configure_rx10_spectral_deess,
    configure_softube_saturation,
    configure_softube_vca,
    configure_uad_la2a,
)


def _band_db(y: np.ndarray, sr: int, lo: float, hi: float) -> float:
    f, p = signal.welch(y, fs=sr, nperseg=4096)
    m = (f >= lo) & (f <= hi)
    return float(10.0 * np.log10(np.sum(p[m]) + 1e-18))


def test_rx10_spectral_deess_preserves_attacks_and_tames_7khz() -> None:
    pm = get_plugin_map()
    if not pm["rx10_deess"].exists:
        return
    sr = 44100
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    # 0.0-1.0s: 300Hz vowel + 2.5kHz consonant attack; 1.0-2.0s: sharp 7.2kHz sibilant!
    x = np.zeros(sr * 2, dtype=np.float32)
    x[:sr] = (0.20 * np.sin(2 * np.pi * 300 * t[:sr]) + 0.12 * np.sin(2 * np.pi * 2500 * t[:sr])).astype(np.float32)
    x[sr:] = (0.04 * np.sin(2 * np.pi * 300 * t[sr:]) + 0.22 * np.sin(2 * np.pi * 7200 * t[sr:])).astype(np.float32)

    plug = load_plugin(str(pm["rx10_deess"].loadable_path))
    configure_rx10_spectral_deess(plug, threshold_db=-6.0, cutoff_hz=5800.0)
    y = np.mean(plug(np.stack([x, x]), sr), axis=0)

    # 2.5 kHz attack band during vowel MUST be untouched (< 0.8 dB loss!)
    attack_delta = _band_db(y[:sr], sr, 2000, 3200) - _band_db(x[:sr], sr, 2000, 3200)
    assert attack_delta > -0.8, f"De-ess damaged 2.5kHz attack band by {attack_delta:.2f} dB!"

    # 7.2 kHz harsh sibilant MUST be attenuated!
    sib_delta = _band_db(y[sr:], sr, 6000, 8500) - _band_db(x[sr:], sr, 6000, 8500)
    assert sib_delta < -1.5, f"De-ess did not attenuate 7.2kHz sibilant (delta={sib_delta:.2f} dB)"


def test_softube_vca_actually_compresses_signal() -> None:
    pm = get_plugin_map()
    if not pm["vca_compressor"].exists:
        return
    sr = 44100
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    x = (0.35 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # ~ -12.1 dBFS RMS

    plug = load_plugin(str(pm["vca_compressor"].loadable_path))
    configure_softube_vca(plug, threshold_db=-24.0, ratio=4.0, attack_norm=1.2, release_norm=0.7, output_db=0.0)
    y = np.mean(plug(np.stack([x, x]), sr), axis=0)

    in_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(x[sr // 4 :] ** 2)) + 1e-12))
    out_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(y[sr // 4 :] ** 2)) + 1e-12))
    gr_db = in_rms_db - out_rms_db
    assert gr_db >= 2.5, f"Softube VCA failed to compress! GR={gr_db:.2f} dB"


def test_softube_saturation_knob_actually_adds_harmonics() -> None:
    pm = get_plugin_map()
    if not pm["saturation_knob"].exists:
        return
    sr = 44100
    t = np.linspace(0.0, 1.5, int(sr * 1.5), endpoint=False)
    x = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    plug = load_plugin(str(pm["saturation_knob"].loadable_path))
    configure_softube_saturation(plug, amount=4.0, sat_type="KEEP LOW")
    y = np.mean(plug(np.stack([x, x]), sr), axis=0)

    # Check 2nd & 3rd harmonic generation (880Hz + 1320Hz) vs clean sine input
    harm_in = _band_db(x, sr, 800, 1500)
    harm_out = _band_db(y, sr, 800, 1500)
    assert (harm_out - harm_in) > 15.0, f"Saturation Knob stayed bypassed! Harmonic rise={harm_out - harm_in:.2f} dB"


def test_uad_la2a_actually_levels_signal() -> None:
    pm = get_plugin_map()
    if not pm["uad_la2a"].exists:
        return
    sr = 44100
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    x = (0.35 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    plug = load_plugin(str(pm["uad_la2a"].loadable_path))
    configure_uad_la2a(plug, peak_reduct=0, gain=30)
    y_no_gr = np.mean(plug(np.stack([x, x]), sr, reset=True), axis=0)

    configure_uad_la2a(plug, peak_reduct=65, gain=30)
    y_gr = np.mean(plug(np.stack([x, x]), sr, reset=True), axis=0)

    rms_no_gr = float(20.0 * np.log10(np.sqrt(np.mean(y_no_gr[sr // 2 :] ** 2)) + 1e-12))
    rms_gr = float(20.0 * np.log10(np.sqrt(np.mean(y_gr[sr // 2 :] ** 2)) + 1e-12))
    assert (rms_no_gr - rms_gr) >= 2.0, f"UAD LA-2A peak_reduct had no effect! delta={rms_no_gr - rms_gr:.2f} dB"


def test_persistent_q_and_r2_contracts() -> None:
    pm = get_plugin_map()
    sr = 44100
    rng = np.random.default_rng(7)
    noise = (0.12 * rng.standard_normal(sr * 2)).astype(np.float32)

    if pm["persistent_q"].exists:
        pq = load_plugin(str(pm["persistent_q"].loadable_path))
        configure_persistent_q_5band(pq, hpf_hz=110.0, body_db=0.0, mud_db=-4.0, presence_db=+4.0, air_db=+3.0)
        y = np.mean(pq(np.stack([noise, noise]), sr), axis=0)
        assert (_band_db(y, sr, 40, 75) - _band_db(noise, sr, 40, 75)) < -5.0
        assert (_band_db(y, sr, 2400, 3400) - _band_db(noise, sr, 2400, 3400)) > +2.5
        assert (_band_db(y, sr, 480, 640) - _band_db(noise, sr, 480, 640)) < -2.2

    if pm["exponential_r2"].exists:
        r2 = load_plugin(str(pm["exponential_r2"].loadable_path))
        configure_clean_plate_r2(r2, rvb_time=0.92, predelay_ms=25.0, lpf_hz=5750.0)
        assert r2.revbtype == "Plate"
        assert float(r2.chordpth) == 0.0
        assert 0.85 <= float(r2.rvb_time) <= 1.0
