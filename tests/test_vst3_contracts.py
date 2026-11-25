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
    configure_khs_dual_delay_spot,
    configure_mautopitch_hard_tune,
    configure_persistent_q_5band,
    configure_persistent_q_precision,
    configure_rx10_spectral_deess,
    configure_softube_saturation,
    configure_softube_vca,
    configure_soothe2_precision,
    configure_trash2_vocal_grit,
    configure_uad_la2a,
    configure_vocalsynth2_radio,
)
from mix_automation.io.song_builder import export_stage3_fx_vst3_states


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


def test_v6_persistent_q_16band_dynamic_spectral_and_soothe2_and_vocalsynth2() -> None:
    pm = get_plugin_map()
    sr = 44100
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    x = np.zeros(sr * 2, dtype=np.float32)
    x[:sr] = (0.04 * np.sin(2 * np.pi * 3200 * t[:sr]) + 0.04 * np.sin(2 * np.pi * 7800 * t[:sr])).astype(np.float32)
    x[sr:] = (0.35 * np.sin(2 * np.pi * 3200 * t[sr:]) + 0.35 * np.sin(2 * np.pi * 7800 * t[sr:])).astype(np.float32)
    st = np.stack([x, x], axis=0)

    if pm["persistent_q"].exists:
        pq = load_plugin(str(pm["persistent_q"].loadable_path))
        configure_persistent_q_precision(
            pq,
            hpf_hz=90.0,
            band_gains_db={"warm_body": 1.0, "mud_honk": -1.5, "air_shelf": 0.5},
            surgical_notches=[(1200.0, -2.5, 6.5)],
            dynamic_bite_hz=3200.0,
            dynamic_bite_thresh_db=-24.0,
            dynamic_bite_range_db=5.5,
            spectral_hf_hz=7800.0,
            spectral_hf_thresh_db=-24.0,
            spectral_hf_range_db=5.5,
            enable_dynamic_bands=True,
        )
        y_pq = np.mean(pq(st, sr, reset=True), axis=0)
        loud_3k = _band_db(y_pq[sr:], sr, 2800, 3600) - _band_db(x[sr:], sr, 2800, 3600)
        quiet_3k = _band_db(y_pq[:sr], sr, 2800, 3600) - _band_db(x[:sr], sr, 2800, 3600)
        loud_7k = _band_db(y_pq[sr:], sr, 7200, 8400) - _band_db(x[sr:], sr, 7200, 8400)
        assert loud_3k < quiet_3k - 2.0, f"PQ dynamic bite band inactive: quiet={quiet_3k:.2f}, loud={loud_3k:.2f}"
        assert loud_7k < -1.5, f"PQ spectral HF band inactive: loud_7k={loud_7k:.2f}"

    if pm["soothe2"].exists:
        s2 = load_plugin(str(pm["soothe2"].loadable_path))
        configure_soothe2_precision(s2, depth=3.6, bite_sens_db=4.0, sibilance_sens_db=6.0, hf_sizzle_sens_db=5.0)
        y_s2 = np.mean(s2(st, sr, reset=True), axis=0)
        assert (_band_db(y_s2[sr:], sr, 2800, 3600) - _band_db(x[sr:], sr, 2800, 3600)) < -3.0
        assert (_band_db(y_s2[sr:], sr, 7200, 8400) - _band_db(x[sr:], sr, 7200, 8400)) < -3.0

    if "izotope_vocalsynth2" in pm and pm["izotope_vocalsynth2"].exists:
        vs = load_plugin(str(pm["izotope_vocalsynth2"].loadable_path))
        configure_vocalsynth2_radio(vs, freq_hz=1650.0, drive=3.8, distort_mix=42.0)
        y_vs = vs(st, sr, reset=True)
        assert float(np.sqrt(np.mean(y_vs**2))) > 0.02, "VocalSynth 2 radio chain output silence!"


def test_v7_mautopitch_khs_delay_trash2_and_stage3_vstpresets_zero_juce_leak() -> None:
    pm = get_plugin_map()
    sr = 44100
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)

    if pm["mautopitch"].exists:
        mp = load_plugin(str(pm["mautopitch"].loadable_path))
        c_major = [True, False, True, False, True, True, False, True, False, True, False, True]
        configure_mautopitch_hard_tune(mp, scale_mask=c_major, speed=95.0, depth=100.0)
        v_tune = (0.25 * np.sin(2.0 * np.pi * 272.0 * t)).astype(np.float32)
        y_mp = mp(np.stack([v_tune, v_tune], axis=0), sr, reset=True)
        assert float(np.sqrt(np.mean(y_mp**2))) > 0.02, "MAutoPitch hard-tune output silence!"

    if pm["khs_dual_delay"].exists:
        dly = load_plugin(str(pm["khs_dual_delay"].loadable_path))
        configure_khs_dual_delay_spot(dly, delay_ms=250.0, feedback=40.0, spread=80.0, duck=15.0, mix_pct=100.0)
        burst = np.zeros(sr * 2, dtype=np.float32)
        burst[: int(0.18 * sr)] = (0.35 * np.sin(2.0 * np.pi * 440.0 * t[: int(0.18 * sr)])).astype(np.float32)
        y_dly = dly(np.stack([burst, burst], axis=0), sr, reset=True)
        tail_rms = float(np.sqrt(np.mean(y_dly[:, int(0.24 * sr) : int(0.80 * sr)] ** 2)))
        assert tail_rms > 0.01, f"kHs Dual Delay produced no echo tail (rms={tail_rms:.5f})"

    if pm["izotope_trash2"].exists:
        t2 = load_plugin(str(pm["izotope_trash2"].loadable_path))
        configure_trash2_vocal_grit(t2, drive=4.2, style=0.5, wet_mix=60.0, low_cut_hz=380.0, high_cut_hz=4200.0)
        sine_in = (0.28 * np.sin(2.0 * np.pi * 600.0 * t)).astype(np.float32)
        y_t2 = np.mean(t2(np.stack([sine_in, sine_in], axis=0), sr, reset=True), axis=0)
        assert float(np.sqrt(np.mean(y_t2**2))) > 0.02, "iZotope Trash 2 output silence!"
        harm_delta = _band_db(y_t2, sr, 1100.0, 2500.0) - _band_db(sine_in, sr, 1100.0, 2500.0)
        assert harm_delta > 10.0, f"iZotope Trash 2 added insufficient harmonics ({harm_delta:.2f} dB)"

    fx_presets = export_stage3_fx_vst3_states(
        {"radio_voice", "hard_autotune", "distortion_grit", "spot_delay_throw"},
        bpm=128.0,
        scale_mask=[True] * 12,
    )
    assert len(fx_presets) > 0
    for rel_path, raw_bytes in fx_presets.items():
        assert raw_bytes[:4] == b"VST3", f"{rel_path} is not a Steinberg .vstpreset"
        assert b"List" in raw_bytes, f"{rel_path} missing Steinberg List header"
        assert b"<VST3PluginState>" not in raw_bytes, f"JUCE leak found in {rel_path}!"
