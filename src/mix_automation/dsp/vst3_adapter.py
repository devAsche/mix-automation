"""Strict Contract-Verified VST3 Parameter Adapter & Ultra-Precision v6 Plugin Helpers.

Eliminates silent parameter failures across all commercial VST3 plugins (`RX 10`, `Persistent Q`
16-band Static+Notch+Dynamic+Spectral EQ, `Oeksound Soothe2` 4-band surgical resonance polish,
`Softube VCA Compressor`, `UAD LA-2A`, `Softube Saturation Knob`, `UAD Pultec`, `TDR SlickEQ`,
`PhoenixVerb`, `Exponential Audio R2`, and newly licensed `iZotope VocalSynth 2`).

Every parameter write is strictly coerced to the exact discrete/string/numeric token advertised by
the VST3 plugin (including Soothe2 kHz notation like `'2k5'`, `'6k5'`, `'10k'`), assigned, and
read back for confirmation.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_KHZ_RE = re.compile(r"^\s*([-+]?\d+)[kK](\d+)?\s*$")


def _parse_numeric_token(token: str) -> float | None:
    s = (
        token.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
        .strip()
        .lower()
    )
    if s == "fast":
        return 0.0
    if s == "slow":
        return 10.0
    mk = _KHZ_RE.match(s)
    if mk:
        whole = mk.group(1)
        frac = mk.group(2)
        return float(f"{whole}.{frac}") * 1000.0 if frac is not None else float(whole) * 1000.0
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def resolve_vst3_value(param: Any, key: str, val: Any) -> Any:
    """Resolve `val` to the exact canonical value accepted by Pedalboard's VST3 parameter."""
    vv = getattr(param, "valid_values", None)
    if not vv:
        return val

    # 1. Exact membership shortcut
    if val in vv:
        return val

    first = vv[0]

    # 2. Boolean parameters
    if isinstance(first, bool):
        return bool(val)

    # 3. Special Wet/Dry alias for Exponential Audio R2 / PhoenixVerb
    if key == "mix" and isinstance(first, str):
        if isinstance(val, (int, float)) and float(val) >= 99.5 and "Wet" in vv:
            return "Wet"
        if isinstance(val, (int, float)) and float(val) <= 0.5 and "Dry" in vv:
            return "Dry"

    # 4. Pure numeric discrete valid_values list (Persistent Q, R2, Soothe2 numeric, RX 10, SlickEQ, VocalSynth 2)
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        target = float(val)
        finite_vv = [x for x in vv if np.isfinite(float(x))]
        pool = finite_vv if finite_vv else vv
        return min(pool, key=lambda x: abs(float(x) - target))

    # 5. String valid_values list (VCA ratio, Saturation Knob, UAD LA-2A, Soothe2 kHz/attack, PhoenixVerb, Pultec)
    if isinstance(first, str):
        if isinstance(val, str):
            norm_target = val.strip().lower()
            for cand in vv:
                if cand.strip().lower() == norm_target:
                    return cand
            num_val = _parse_numeric_token(val)
            if num_val is None:
                raise ValueError(f"String value {val!r} not in valid_values for {key!r}: {vv[:8]}")
        else:
            num_val = float(val)

        numeric_cands: list[tuple[float, str]] = []
        for cand in vv:
            parsed = _parse_numeric_token(cand)
            if parsed is not None:
                numeric_cands.append((parsed, cand))
        if not numeric_cands:
            raise ValueError(f"Cannot map numeric {val!r} to string parameter {key!r}: {vv[:8]}")
        return min(numeric_cands, key=lambda pair: abs(pair[0] - num_val))[1]

    return val


def set_vst3_param_strict(plug: Any, key: str, val: Any) -> Any:
    """Strictly assign a VST3 parameter with canonical resolution and read-back verification."""
    if plug is None:
        raise ValueError(f"Plugin instance is None when setting {key!r}.")
    if key not in plug.parameters:
        raise KeyError(f"Parameter {key!r} does not exist on plugin {plug}.")

    param = plug.parameters[key]
    resolved = resolve_vst3_value(param, key, val)
    setattr(plug, key, resolved)

    actual = getattr(plug, key)
    if isinstance(resolved, str):
        if str(actual).strip().lower() != resolved.strip().lower():
            raise RuntimeError(f"VST3 write verification failed on {key!r}: wrote {resolved!r}, read {actual!r}")
    elif isinstance(resolved, bool):
        if bool(actual) != resolved:
            raise RuntimeError(f"VST3 write verification failed on {key!r}: wrote {resolved!r}, read {actual!r}")
    elif isinstance(resolved, (int, float)) and np.isfinite(float(resolved)):
        if abs(float(actual) - float(resolved)) > max(1e-2, abs(float(resolved)) * 0.02):
            raise RuntimeError(f"VST3 write verification failed on {key!r}: wrote {resolved!r}, read {actual!r}")
    return actual


# --- Verified High-Level Plugin Configurators ---


def configure_rx10_voice_denoise(plug: Any, reduction_db: float = 4.2) -> None:
    """Configure RX 10 Voice De-noise specifically for singing vocals ('Music' + 'Gentle')."""
    set_vst3_param_strict(plug, "global_bypass", False)
    set_vst3_param_strict(plug, "optimize_for", "Music")
    set_vst3_param_strict(plug, "filter_type", "Gentle")
    set_vst3_param_strict(plug, "adaptive_mode", True)
    set_vst3_param_strict(plug, "reduction", float(np.clip(reduction_db, 0.0, 8.0)))


def configure_rx10_spectral_deess(
    plug: Any,
    threshold_db: float = -6.0,
    cutoff_hz: float = 5800.0,
    spectral_shaping: float = 45.0,
) -> None:
    """Configure RX 10 De-ess in surgical Spectral mode strictly ABOVE 5.4 kHz (protects attacks!)."""
    set_vst3_param_strict(plug, "global_bypass", False)
    set_vst3_param_strict(plug, "output_ess_only", False)
    set_vst3_param_strict(plug, "algorithm", "Spectral De-ess")
    set_vst3_param_strict(plug, "absolute_mode", False)
    set_vst3_param_strict(plug, "cutoff_freq", float(np.clip(cutoff_hz, 5200.0, 7800.0)))
    set_vst3_param_strict(plug, "threshold", float(np.clip(threshold_db, -18.0, 0.0)))
    set_vst3_param_strict(plug, "spectral_shaping", float(np.clip(spectral_shaping, 0.0, 100.0)))
    set_vst3_param_strict(plug, "speed", "Fast")


# Canonical 10-band ERB/Octave vocal sculpting frequencies (Hz) & Q factors for v6 Precision EQ
PRECISION_EQ_BAND_SPECS: tuple[tuple[str, float, float, str], ...] = (
    ("low_chest", 160.0, 1.15, "Peaking"),
    ("warm_body", 260.0, 1.25, "Peaking"),
    ("low_box", 420.0, 1.35, "Peaking"),
    ("mud_honk", 680.0, 1.40, "Peaking"),
    ("throat_core", 1050.0, 1.35, "Peaking"),
    ("upper_core", 1600.0, 1.35, "Peaking"),
    ("vocal_bite", 2400.0, 1.30, "Peaking"),
    ("clarity_pres", 3600.0, 1.25, "Peaking"),
    ("sibilant_sheen", 5800.0, 1.15, "Peaking"),
    ("air_shelf", 10500.0, 0.72, "HighShelf"),
)


def configure_persistent_q_precision(
    plug: Any,
    hpf_hz: float,
    band_gains_db: dict[str, float],
    surgical_notches: list[tuple[float, float, float]] | None = None,
    dynamic_mud_thresh_db: float = -24.0,
    dynamic_mud_range_db: float = 3.2,
    dynamic_bite_hz: float = 3250.0,
    dynamic_bite_thresh_db: float = -23.0,
    dynamic_bite_range_db: float = 4.0,
    spectral_hf_hz: float = 7800.0,
    spectral_hf_thresh_db: float = -28.0,
    spectral_hf_range_db: float = 4.5,
    enable_dynamic_bands: bool = True,
) -> None:
    """Configure Persistent Q as a 16-Band Ultra-Precision Static + Surgical Notch + Dynamic + Spectral EQ.

    Layout (all 16 bands serialize natively 1:1 into Studio One 5!):
      - Band 1     : F0-adaptive 18 dB/oct Highpass Filter
      - Bands 2-11 : 10-band ERB/Octave differentiable vocal sculpting EQ (160 Hz .. 10.5 kHz)
      - Bands 12-13: Automated narrow-Q surgical resonance notches (`Q = 5.5..8.0`)
      - Band 14    : Dynamic low-mid proximity/mud tamer (`290 Hz`, `dynamic=True`)
      - Band 15    : Ultra-fast Dynamic harsh-attack / consonant-bite tamer (`attack=1.2ms`, `dynamic=True`)
      - Band 16    : Real-time Spectral HF sizzle & sibilance silkener (`spectral=True`)
    """
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "mix", 100.0)

    # Disable unused bands cleanly first (bands 1..18) so zero stale state carries over
    for b_idx in range(1, 19):
        en_k = f"band_{b_idx}_enabled"
        if en_k in plug.parameters:
            set_vst3_param_strict(plug, en_k, False)
        dyn_k = f"band_{b_idx}_dynamic"
        if dyn_k in plug.parameters:
            set_vst3_param_strict(plug, dyn_k, False)
        spec_k = f"band_{b_idx}_spectral"
        if spec_k in plug.parameters:
            set_vst3_param_strict(plug, spec_k, False)

    # Band 1: F0-Adaptive Highpass
    set_vst3_param_strict(plug, "band_1_enabled", True)
    set_vst3_param_strict(plug, "band_1_type", "Highpass")
    set_vst3_param_strict(plug, "band_1_frequency", float(np.clip(hpf_hz, 60.0, 160.0)))
    set_vst3_param_strict(plug, "band_1_slope", 18.0)

    # Bands 2-11: 10-Band Differentiable Sculpting EQ
    for offset, (b_name, freq_hz, q_val, b_type) in enumerate(PRECISION_EQ_BAND_SPECS, start=2):
        g_val = float(np.clip(band_gains_db.get(b_name, 0.0), -4.5, 4.0))
        if b_name == "air_shelf":
            g_val = float(np.clip(g_val, -4.0, 2.2))  # Guardrail: never blast home-mic hiss!
        set_vst3_param_strict(plug, f"band_{offset}_enabled", True)
        set_vst3_param_strict(plug, f"band_{offset}_type", b_type)
        set_vst3_param_strict(plug, f"band_{offset}_frequency", float(freq_hz))
        set_vst3_param_strict(plug, f"band_{offset}_gain", g_val)
        set_vst3_param_strict(plug, f"band_{offset}_q", float(q_val))

    # Bands 12-13: Automated Narrow-Q Surgical Resonance Notches (up to 2 detected resonances)
    notches = (surgical_notches or [])[:2]
    for n_idx, (n_freq, n_gain, n_q) in enumerate(notches, start=12):
        if abs(n_gain) < 0.25:
            continue
        set_vst3_param_strict(plug, f"band_{n_idx}_enabled", True)
        set_vst3_param_strict(plug, f"band_{n_idx}_type", "Peaking")
        set_vst3_param_strict(plug, f"band_{n_idx}_frequency", float(np.clip(n_freq, 220.0, 8500.0)))
        set_vst3_param_strict(plug, f"band_{n_idx}_gain", float(np.clip(n_gain, -4.2, -0.3)))
        set_vst3_param_strict(plug, f"band_{n_idx}_q", float(np.clip(n_q, 4.0, 10.0)))

    if enable_dynamic_bands:
        # Band 14: Dynamic Proximity / Low-Mid Mud Control (~290 Hz)
        if dynamic_mud_range_db > 0.2:
            set_vst3_param_strict(plug, "band_14_enabled", True)
            set_vst3_param_strict(plug, "band_14_type", "Peaking")
            set_vst3_param_strict(plug, "band_14_frequency", 290.0)
            set_vst3_param_strict(plug, "band_14_gain", 0.0)
            set_vst3_param_strict(plug, "band_14_q", 1.35)
            set_vst3_param_strict(plug, "band_14_dynamic", True)
            set_vst3_param_strict(plug, "band_14_threshold", float(np.clip(dynamic_mud_thresh_db, -48.0, -8.0)))
            set_vst3_param_strict(plug, "band_14_dynamic_range", float(np.clip(dynamic_mud_range_db, 0.5, 6.0)))
            set_vst3_param_strict(plug, "band_14_ratio", 2.6)
            set_vst3_param_strict(plug, "band_14_knee", 10.0)
            set_vst3_param_strict(plug, "band_14_attack", 14.0)
            set_vst3_param_strict(plug, "band_14_release", 135.0)

        # Band 15: Ultra-Fast Dynamic Harsh Consonant Attack & Upper-Mid Bite Tamer (2.8-4.2 kHz)
        if dynamic_bite_range_db > 0.2:
            set_vst3_param_strict(plug, "band_15_enabled", True)
            set_vst3_param_strict(plug, "band_15_type", "Peaking")
            set_vst3_param_strict(plug, "band_15_frequency", float(np.clip(dynamic_bite_hz, 2400.0, 4400.0)))
            set_vst3_param_strict(plug, "band_15_gain", 0.0)
            set_vst3_param_strict(plug, "band_15_q", 1.50)
            set_vst3_param_strict(plug, "band_15_dynamic", True)
            set_vst3_param_strict(plug, "band_15_threshold", float(np.clip(dynamic_bite_thresh_db, -46.0, -8.0)))
            set_vst3_param_strict(plug, "band_15_dynamic_range", float(np.clip(dynamic_bite_range_db, 0.5, 7.0)))
            set_vst3_param_strict(plug, "band_15_ratio", 3.4)
            set_vst3_param_strict(plug, "band_15_knee", 8.0)
            set_vst3_param_strict(plug, "band_15_attack", 1.2)
            set_vst3_param_strict(plug, "band_15_release", 55.0)

        # Band 16: Real-Time Spectral Dynamic HF Sizzle & Sibilance Silkener (6.5-10 kHz)
        if spectral_hf_range_db > 0.2:
            set_vst3_param_strict(plug, "band_16_enabled", True)
            set_vst3_param_strict(plug, "band_16_type", "Peaking")
            set_vst3_param_strict(plug, "band_16_frequency", float(np.clip(spectral_hf_hz, 6000.0, 10500.0)))
            set_vst3_param_strict(plug, "band_16_gain", 0.0)
            set_vst3_param_strict(plug, "band_16_q", 1.05)
            set_vst3_param_strict(plug, "band_16_spectral", True)
            set_vst3_param_strict(plug, "band_16_spectral_threshold", float(np.clip(spectral_hf_thresh_db, -54.0, -10.0)))
            set_vst3_param_strict(plug, "band_16_spectral_range", float(np.clip(spectral_hf_range_db, 0.5, 8.0)))
            set_vst3_param_strict(plug, "band_16_spectral_ratio", 3.5)
            set_vst3_param_strict(plug, "band_16_spectral_density", 65.0)
            set_vst3_param_strict(plug, "band_16_spectral_attack", 1.5)
            set_vst3_param_strict(plug, "band_16_spectral_release", 50.0)


def configure_persistent_q_harmony_hollow(
    plug: Any,
    lead_core_hz: float = 1550.0,
    hpf_hz: float = 240.0,
    lpf_hz: float = 8500.0,
) -> None:
    """Configure Persistent Q with Pro Hollow-Core Harmony EQ + Dynamic Sibilance Control.

    Carves out the lead vocal's central formant core (`lead_core_hz` & `2.8 kHz`) so harmonies
    never fight the lead's presence, while preserving warm harmonic chord richness (`380-550 Hz`)
    and taming harmony sibilants dynamically (`6.5 kHz`).
    """
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "mix", 100.0)
    for b_idx in range(1, 10):
        if f"band_{b_idx}_enabled" in plug.parameters:
            set_vst3_param_strict(plug, f"band_{b_idx}_enabled", False)
        if f"band_{b_idx}_dynamic" in plug.parameters:
            set_vst3_param_strict(plug, f"band_{b_idx}_dynamic", False)
        if f"band_{b_idx}_spectral" in plug.parameters:
            set_vst3_param_strict(plug, f"band_{b_idx}_spectral", False)

    # Band 1: Clean Highpass (keeps low end out of the lead's chest)
    set_vst3_param_strict(plug, "band_1_enabled", True)
    set_vst3_param_strict(plug, "band_1_type", "Highpass")
    set_vst3_param_strict(plug, "band_1_frequency", float(np.clip(hpf_hz, 180.0, 320.0)))
    set_vst3_param_strict(plug, "band_1_slope", 18.0)

    # Band 2: Warm chordal harmony richness (420 Hz gentle lift)
    set_vst3_param_strict(plug, "band_2_enabled", True)
    set_vst3_param_strict(plug, "band_2_type", "Peaking")
    set_vst3_param_strict(plug, "band_2_frequency", 420.0)
    set_vst3_param_strict(plug, "band_2_gain", +1.0)
    set_vst3_param_strict(plug, "band_2_q", 1.1)

    # Band 3: Hollow-Core Cut centered on lead vocal's central formant ("声の芯を抜く")
    set_vst3_param_strict(plug, "band_3_enabled", True)
    set_vst3_param_strict(plug, "band_3_type", "Peaking")
    set_vst3_param_strict(plug, "band_3_frequency", float(np.clip(lead_core_hz, 1050.0, 2200.0)))
    set_vst3_param_strict(plug, "band_3_gain", -4.4)
    set_vst3_param_strict(plug, "band_3_q", 1.05)

    # Band 4: Lead Presence Pocket Cut (2.85 kHz)
    set_vst3_param_strict(plug, "band_4_enabled", True)
    set_vst3_param_strict(plug, "band_4_type", "Peaking")
    set_vst3_param_strict(plug, "band_4_frequency", 2850.0)
    set_vst3_param_strict(plug, "band_4_gain", -3.8)
    set_vst3_param_strict(plug, "band_4_q", 1.10)

    # Band 5: Gentle Lowpass smoothing harsh top edge
    set_vst3_param_strict(plug, "band_5_enabled", True)
    set_vst3_param_strict(plug, "band_5_type", "Lowpass")
    set_vst3_param_strict(plug, "band_5_frequency", float(np.clip(lpf_hz, 6500.0, 10500.0)))
    set_vst3_param_strict(plug, "band_5_slope", 12.0)

    # Band 6: Dynamic Harmony Sibilance Melt (6.4 kHz)
    set_vst3_param_strict(plug, "band_6_enabled", True)
    set_vst3_param_strict(plug, "band_6_type", "Peaking")
    set_vst3_param_strict(plug, "band_6_frequency", 6400.0)
    set_vst3_param_strict(plug, "band_6_gain", -1.2)
    set_vst3_param_strict(plug, "band_6_q", 1.15)
    set_vst3_param_strict(plug, "band_6_dynamic", True)
    set_vst3_param_strict(plug, "band_6_threshold", -32.0)
    set_vst3_param_strict(plug, "band_6_dynamic_range", 5.5)
    set_vst3_param_strict(plug, "band_6_ratio", 4.0)
    set_vst3_param_strict(plug, "band_6_attack", 0.8)
    set_vst3_param_strict(plug, "band_6_release", 50.0)


def configure_persistent_q_5band(
    plug: Any,
    hpf_hz: float,
    body_db: float,
    mud_db: float,
    presence_db: float,
    air_db: float,
    body_hz: float = 225.0,
    mud_hz: float = 560.0,
    presence_hz: float = 2900.0,
    air_hz: float = 10500.0,
    mid_dip_hz: float | None = None,
    mid_dip_db: float = 0.0,
) -> None:
    """Backwards-compatible static 5-or-6-band Persistent Q configurator."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "mix", 100.0)

    set_vst3_param_strict(plug, "band_1_enabled", True)
    set_vst3_param_strict(plug, "band_1_type", "Highpass")
    set_vst3_param_strict(plug, "band_1_frequency", float(np.clip(hpf_hz, 55.0, 180.0)))
    set_vst3_param_strict(plug, "band_1_slope", 18.0)
    set_vst3_param_strict(plug, "band_1_dynamic", False)

    set_vst3_param_strict(plug, "band_2_enabled", True)
    set_vst3_param_strict(plug, "band_2_type", "Peaking")
    set_vst3_param_strict(plug, "band_2_frequency", float(body_hz))
    set_vst3_param_strict(plug, "band_2_gain", float(np.clip(body_db, -6.0, 6.0)))
    set_vst3_param_strict(plug, "band_2_q", 0.90)
    set_vst3_param_strict(plug, "band_2_dynamic", False)

    set_vst3_param_strict(plug, "band_3_enabled", True)
    set_vst3_param_strict(plug, "band_3_type", "Peaking")
    set_vst3_param_strict(plug, "band_3_frequency", float(mud_hz))
    set_vst3_param_strict(plug, "band_3_gain", float(np.clip(mud_db, -7.0, 3.0)))
    set_vst3_param_strict(plug, "band_3_q", 1.25)
    set_vst3_param_strict(plug, "band_3_dynamic", False)

    set_vst3_param_strict(plug, "band_4_enabled", True)
    set_vst3_param_strict(plug, "band_4_type", "Peaking")
    set_vst3_param_strict(plug, "band_4_frequency", float(presence_hz))
    set_vst3_param_strict(plug, "band_4_gain", float(np.clip(presence_db, -4.0, 6.5)))
    set_vst3_param_strict(plug, "band_4_q", 0.80)
    set_vst3_param_strict(plug, "band_4_dynamic", False)

    set_vst3_param_strict(plug, "band_5_enabled", True)
    set_vst3_param_strict(plug, "band_5_type", "HighShelf")
    set_vst3_param_strict(plug, "band_5_frequency", float(air_hz))
    set_vst3_param_strict(plug, "band_5_gain", float(np.clip(air_db, -4.0, 6.5)))
    set_vst3_param_strict(plug, "band_5_q", 0.70)
    set_vst3_param_strict(plug, "band_5_dynamic", False)

    if mid_dip_hz is not None and abs(mid_dip_db) > 0.05:
        set_vst3_param_strict(plug, "band_6_enabled", True)
        set_vst3_param_strict(plug, "band_6_type", "Peaking")
        set_vst3_param_strict(plug, "band_6_frequency", float(mid_dip_hz))
        set_vst3_param_strict(plug, "band_6_gain", float(np.clip(mid_dip_db, -5.0, 5.0)))
        set_vst3_param_strict(plug, "band_6_q", 1.10)
        set_vst3_param_strict(plug, "band_6_dynamic", False)
    elif "band_6_enabled" in plug.parameters:
        set_vst3_param_strict(plug, "band_6_enabled", False)


def configure_soothe2_precision(
    plug: Any,
    depth: float = 3.2,
    sharpness: float = 5.2,
    selectivity: float = 4.4,
    attack: float = 0.7,
    release: float = 1.8,
    low_cut_hz: float = 380.0,
    high_cut_hz: float = 16500.0,
    bite_sens_db: float = 3.8,
    sibilance_sens_db: float = 5.8,
    hf_sizzle_sens_db: float = 4.8,
    mode: str = "soft",
    resolution: str = "high",
) -> None:
    """Configure Oeksound Soothe2 as a surgical Post-Comp / Post-Sat harshness & HF-sizzle polisher."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "delta", False)
    set_vst3_param_strict(plug, "mix", 100.0)
    set_vst3_param_strict(plug, "mode", mode)
    set_vst3_param_strict(plug, "depth", float(np.clip(depth, 0.2, 8.0)))
    set_vst3_param_strict(plug, "sharpness", float(np.clip(sharpness, 2.0, 9.0)))
    set_vst3_param_strict(plug, "selectivity", float(np.clip(selectivity, 1.5, 8.5)))
    set_vst3_param_strict(plug, "attack", float(np.clip(attack, 0.0, 10.0)))
    set_vst3_param_strict(plug, "release", float(np.clip(release, 0.0, 10.0)))
    set_vst3_param_strict(plug, "resolution", resolution)

    set_vst3_param_strict(plug, "low_cut_on", True)
    set_vst3_param_strict(plug, "low_cut_freq_hz", float(np.clip(low_cut_hz, 200.0, 1200.0)))
    set_vst3_param_strict(plug, "high_cut_on", True)
    set_vst3_param_strict(plug, "high_cut_freq_hz", float(np.clip(high_cut_hz, 10000.0, 20000.0)))

    # Band 1: Protect vocal chest/core (gentle negative sensitivity @ 950 Hz)
    set_vst3_param_strict(plug, "band1_on", True)
    set_vst3_param_strict(plug, "band1_mode", "bell")
    set_vst3_param_strict(plug, "band1_freq_hz", 950.0)
    set_vst3_param_strict(plug, "band1_q", 1.0)
    set_vst3_param_strict(plug, "band1_sens_db", -2.0)

    # Band 2: Upper-mid harsh bite & piercing consonant needle suppression (3.1 kHz)
    set_vst3_param_strict(plug, "band2_on", True)
    set_vst3_param_strict(plug, "band2_mode", "bell")
    set_vst3_param_strict(plug, "band2_freq_hz", 3100.0)
    set_vst3_param_strict(plug, "band2_q", 1.2)
    set_vst3_param_strict(plug, "band2_sens_db", float(np.clip(bite_sens_db, -6.0, 10.0)))

    # Band 3: Sibilant grit & fricative harshness melting (6.4 kHz)
    set_vst3_param_strict(plug, "band3_on", True)
    set_vst3_param_strict(plug, "band3_mode", "bell")
    set_vst3_param_strict(plug, "band3_freq_hz", 6400.0)
    set_vst3_param_strict(plug, "band3_q", 1.1)
    set_vst3_param_strict(plug, "band3_sens_db", float(np.clip(sibilance_sens_db, -6.0, 11.0)))

    # Band 4: High-frequency "ジリジリ" sizzle high-shelf suppression (10 kHz+)
    set_vst3_param_strict(plug, "band4_on", True)
    set_vst3_param_strict(plug, "band4_mode", "high shelf")
    set_vst3_param_strict(plug, "band4_freq_hz", 10000.0)
    set_vst3_param_strict(plug, "band4_q", 0.8)
    set_vst3_param_strict(plug, "band4_sens_db", float(np.clip(hf_sizzle_sens_db, -6.0, 11.0)))


def configure_vocalsynth2_radio(
    plug: Any,
    freq_hz: float = 1650.0,
    drive: float = 3.8,
    distort_mix: float = 42.0,
) -> None:
    """Configure newly licensed iZotope VocalSynth 2 for rich telephone/radio/megaphone vocal FX."""
    set_vst3_param_strict(plug, "global_bypass", False)
    for mod in ("bvx_enable", "plyvx_enable", "vcdr_enable", "tlkbx_enable", "cmpvx_enable", "chrs_enable", "dly_enable", "shrd_enable", "rngmd_enable"):
        if mod in plug.parameters:
            set_vst3_param_strict(plug, mod, False)
    # Route clean dry input into VocalSynth 2 FX bus (`input_lvl_dry_corrected = 100.0`)!
    if "input_lvl_dry_corrected" in plug.parameters:
        set_vst3_param_strict(plug, "input_lvl_dry_corrected", 100.0)
    set_vst3_param_strict(plug, "fltr_enable", True)
    set_vst3_param_strict(plug, "fltr_freq_hz", float(np.clip(freq_hz, 800.0, 3200.0)))
    set_vst3_param_strict(plug, "fltr_res", 26.0)
    set_vst3_param_strict(plug, "fltr_mix", 85.0)
    set_vst3_param_strict(plug, "dstrt_enable", True)
    set_vst3_param_strict(plug, "dstrt_drive", float(np.clip(drive, 0.5, 10.0)))
    set_vst3_param_strict(plug, "dstrt_tone", 54.0)
    set_vst3_param_strict(plug, "dstrt_mix", float(np.clip(distort_mix, 0.0, 100.0)))
    set_vst3_param_strict(plug, "output_mix", 100.0)


def configure_vocalsynth2_biovox_gloss(
    plug: Any,
    wet_mix: float = 14.0,
    breath: float = 18.0,
    clarity: float = 78.0,
    nasal: float = 0.0,
    formant_shift: float = 0.0,
) -> None:
    """Configure licensed iZotope VocalSynth 2 Biovox vocal-tract engine for subtle harmonic gloss."""
    set_vst3_param_strict(plug, "global_bypass", False)
    for mod in ("plyvx_enable", "vcdr_enable", "tlkbx_enable", "cmpvx_enable", "fltr_enable", "dstrt_enable", "dly_enable", "shrd_enable", "rngmd_enable"):
        if mod in plug.parameters:
            set_vst3_param_strict(plug, mod, False)
    if "input_lvl_dry_corrected" in plug.parameters:
        set_vst3_param_strict(plug, "input_lvl_dry_corrected", 75.0)
    set_vst3_param_strict(plug, "bvx_enable", True)
    set_vst3_param_strict(plug, "bvx_lvl", 100.0)
    set_vst3_param_strict(plug, "bvx_breath", float(np.clip(breath, 0.0, 100.0)))
    set_vst3_param_strict(plug, "bvx_clarity", float(np.clip(clarity, 0.0, 100.0)))
    set_vst3_param_strict(plug, "bvx_nasal", float(np.clip(nasal, 0.0, 100.0)))
    set_vst3_param_strict(plug, "bvx_shift", float(np.clip(formant_shift, -50.0, 50.0)))
    set_vst3_param_strict(plug, "bvx_hpf_hz", 260.0)
    set_vst3_param_strict(plug, "bvx_lpf_hz", 7200.0)
    set_vst3_param_strict(plug, "chrs_enable", True)
    set_vst3_param_strict(plug, "chrs_depth", 32.0)
    set_vst3_param_strict(plug, "chrs_width", 80.0)
    set_vst3_param_strict(plug, "chrs_mix", 45.0)
    set_vst3_param_strict(plug, "output_mix", float(np.clip(wet_mix, 0.0, 40.0)))


def configure_softube_vca(
    plug: Any,
    threshold_db: float,
    ratio: float = 3.2,
    attack_norm: float = 0.55,
    release_norm: float = 0.65,
    sc_hpf_hz: float = 120.0,
    output_db: float = 0.0,
    dry_wet: float = 100.0,
    output_drive: float = 0.1,
) -> None:
    """Configure Softube VCA Compressor using its verified negative threshold & string ratio schema."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "compression_db", float(np.clip(threshold_db, -52.0, 10.0)))
    set_vst3_param_strict(plug, "ratio", float(np.clip(ratio, 1.1, 20.0)))
    set_vst3_param_strict(plug, "attack", float(np.clip(attack_norm, 0.25, 4.0)))
    set_vst3_param_strict(plug, "release", float(np.clip(release_norm, 0.25, 4.0)))
    set_vst3_param_strict(plug, "sidechain_filter_hz", float(sc_hpf_hz) if sc_hpf_hz > 20.0 else "Off")
    set_vst3_param_strict(plug, "output_db", float(np.clip(output_db, -24.0, 24.0)))
    set_vst3_param_strict(plug, "output_drive", float(np.clip(output_drive, 0.0, 10.0)))
    set_vst3_param_strict(plug, "dry_wet_wet", float(np.clip(dry_wet, 0.0, 100.0)))


def configure_uad_la2a(
    plug: Any,
    peak_reduct: float = 35.0,
    gain: float = 35.0,
    mode: str = "Comp",
) -> None:
    """Configure UAD Teletronix LA-2A using its verified string integer knobs (`'0'..'100'`)."""
    set_vst3_param_strict(plug, "power", True)
    set_vst3_param_strict(plug, "master_bypass", False)
    set_vst3_param_strict(plug, "comp_limit", mode)
    set_vst3_param_strict(plug, "peak_reduct", int(round(np.clip(peak_reduct, 0.0, 100.0))))
    set_vst3_param_strict(plug, "gain", int(round(np.clip(gain, 0.0, 100.0))))


def configure_softube_saturation(
    plug: Any,
    amount: float = 1.6,
    sat_type: str = "KEEP LOW",
) -> None:
    """Configure Softube Saturation Knob using its verified string values (`'0.0'..'10.0'`, uppercase type)."""
    set_vst3_param_strict(plug, "bypass", False)
    if amount <= 0.05:
        set_vst3_param_strict(plug, "saturation", "OFF")
    else:
        set_vst3_param_strict(plug, "saturation", round(float(np.clip(amount, 0.1, 10.0)), 1))
    set_vst3_param_strict(plug, "saturation_type", sat_type.upper())


def configure_uad_pultec(
    plug: Any,
    lf_boost: float = 0.0,
    hf_boost: float = 0.0,
    lf_freq: str = "100 CPS",
    hf_freq: str = "12 KCS",
    hf_q: float = 4.5,
) -> None:
    """Configure UAD Pultec EQP-1A for vocal warmth & gentle top sheen."""
    set_vst3_param_strict(plug, "power", True)
    set_vst3_param_strict(plug, "master_bypass", False)
    set_vst3_param_strict(plug, "enable", "In")
    set_vst3_param_strict(plug, "low_freq", lf_freq)
    set_vst3_param_strict(plug, "lf_boost", float(np.clip(lf_boost, 0.0, 6.0)))
    set_vst3_param_strict(plug, "lf_atten", 0.0)
    set_vst3_param_strict(plug, "high_freq", hf_freq)
    set_vst3_param_strict(plug, "hf_boost", float(np.clip(hf_boost, 0.0, 7.0)))
    set_vst3_param_strict(plug, "hf_q", float(np.clip(hf_q, 0.0, 10.0)))
    set_vst3_param_strict(plug, "hf_atten", 0.0)


def configure_slick_eq(
    plug: Any,
    out_stage: str = "Silky",
    drive_db: float = 3.5,
    eq_model: str = "American",
    high_gain_db: float = 0.0,
) -> None:
    """Configure TDR VOS SlickEQ analog stage & sheen."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "eq_model", eq_model)
    set_vst3_param_strict(plug, "out_stage", out_stage)
    set_vst3_param_strict(plug, "eq_sat", True)
    set_vst3_param_strict(plug, "out_drive_db", float(np.clip(drive_db, 0.0, 14.0)))
    set_vst3_param_strict(plug, "high_gain_db", float(np.clip(high_gain_db, -4.0, 4.0)))


def configure_waves_mv2(
    plug: Any,
    low_level: float = 4.0,
    high_level: float = -4.0,
    output_gain: float = 0.0,
) -> None:
    """Configure Waves MV2 upward/downward vocal leveler (strictly bounded to avoid noise lift)."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "low_level", float(np.clip(low_level, 0.0, 8.0)))
    set_vst3_param_strict(plug, "high_level", float(np.clip(high_level, -24.0, 0.0)))
    set_vst3_param_strict(plug, "output_gain", float(np.clip(output_gain, -12.0, 0.0)))


def configure_clean_plate_r2(
    plug: Any,
    rvb_time: float = 0.95,
    predelay_ms: float = 24.0,
    lpf_hz: float = 5400.0,
) -> None:
    """Configure Exponential Audio R2 to a strictly chorus-free short studio Plate."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "mix", 100.0)
    set_vst3_param_strict(plug, "revbtype", "Plate")
    set_vst3_param_strict(plug, "rvb_time", float(np.clip(rvb_time, 0.55, 1.8)))
    set_vst3_param_strict(plug, "predelay", float(np.clip(predelay_ms, 4.0, 55.0)))
    set_vst3_param_strict(plug, "chordpth", 0.0)
    set_vst3_param_strict(plug, "reverbfc", float(lpf_hz))
    set_vst3_param_strict(plug, "earlyffc", float(lpf_hz))


def configure_early_room_phoenix(
    plug: Any,
    size_m: float = 8.0,
    rvb_time: float = 0.44,
    predelay_ms: float = 10.0,
) -> None:
    """Configure Exponential Audio PhoenixVerb as a tight studio Early-Reflection room glue."""
    set_vst3_param_strict(plug, "mix", 100.0)
    set_vst3_param_strict(plug, "revbtype", "Chamber")
    set_vst3_param_strict(plug, "rvb_size", float(np.clip(size_m, 4.0, 16.0)))
    set_vst3_param_strict(plug, "rvb_time", float(np.clip(rvb_time, 0.28, 0.75)))
    set_vst3_param_strict(plug, "predelay", float(np.clip(predelay_ms, 3.0, 30.0)))
    set_vst3_param_strict(plug, "earlylvl", 0.0)
    set_vst3_param_strict(plug, "damp_fc", 5200.0)


def configure_mautopitch_hard_tune(
    plug: Any,
    scale_mask: list[bool] | np.ndarray | None = None,
    speed: float = 95.0,
    depth: float = 100.0,
    formant_shift: float = 0.0,
) -> None:
    """Configure MeldaProduction MAutoPitch for crisp Stage-3 hard-tune / robotic pitch quantization."""
    if "bypass" in plug.parameters:
        bp_vv = getattr(plug.parameters["bypass"], "valid_values", None) or []
        if "Off" in bp_vv:
            set_vst3_param_strict(plug, "bypass", "Off")
        elif False in bp_vv:
            set_vst3_param_strict(plug, "bypass", False)

    set_vst3_param_strict(plug, "depth_automatic_tuning", float(np.clip(depth, 0.0, 100.0)))
    set_vst3_param_strict(plug, "speed_automatic_tuning", float(np.clip(speed, 0.0, 100.0)))
    set_vst3_param_strict(plug, "keep_formants_effects", 100.0)
    set_vst3_param_strict(plug, "formant_shift_effects", float(np.clip(formant_shift, -12.0, 12.0)))
    set_vst3_param_strict(plug, "dry_wet_effects", "100% wet, 0% dry")

    note_keys = ("c", "c_sharp", "d", "d_sharp", "e", "f", "f_sharp", "g", "g_sharp", "a", "a_sharp", "b")
    if scale_mask is not None and len(scale_mask) == 12:
        bool_mask = [bool(x) for x in scale_mask]
        if not any(bool_mask):
            bool_mask = [True] * 12
    else:
        bool_mask = [True] * 12

    for nk, enabled in zip(note_keys, bool_mask):
        set_vst3_param_strict(plug, nk, enabled)


def configure_khs_dual_delay_spot(
    plug: Any,
    delay_ms: float = 250.0,
    feedback: float = 36.0,
    spread: float = 75.0,
    duck: float = 25.0,
    mix_pct: float = 100.0,
) -> None:
    """Configure Kilohearts Dual Delay for rhythmic Stage-3 spot delay throws."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "sync_mode", "time")
    set_vst3_param_strict(plug, "delay_time_seconds", float(np.clip(delay_ms, 50.0, 900.0)))
    set_vst3_param_strict(plug, "feedback", float(np.clip(feedback, 0.0, 95.0)))
    set_vst3_param_strict(plug, "crosstalk", 40.0)
    set_vst3_param_strict(plug, "spread", float(np.clip(spread, 0.0, 100.0)))
    set_vst3_param_strict(plug, "duck", float(np.clip(duck, 0.0, 100.0)))
    set_vst3_param_strict(plug, "mix", float(np.clip(mix_pct, 0.0, 100.0)))


def configure_trash2_vocal_grit(
    plug: Any,
    drive: float = 3.6,
    style: float = 0.5,
    wet_mix: float = 55.0,
    low_cut_hz: float = 380.0,
    high_cut_hz: float = 4200.0,
) -> None:
    """Configure iZotope Trash 2 strictly band-limited (380-4200 Hz) for zero-sizzle vocal grit."""
    set_vst3_param_strict(plug, "global_bypass", False)
    set_vst3_param_strict(plug, "trash_module_bypass", False)
    set_vst3_param_strict(plug, "trash_bypass", False)
    if "gain_input_lock" in plug.parameters:
        set_vst3_param_strict(plug, "gain_input_lock", True)
        set_vst3_param_strict(plug, "gain_input_locked_db", float(np.clip(4.0 + drive * 0.8, 2.0, 9.5)))
    set_vst3_param_strict(plug, "trash_stg1_enable", True)
    set_vst3_param_strict(plug, "trash_stg1_shape_enable", True)
    # Drive `-18 LUFS` studio vocals cleanly into Trash 2's nonlinear knee with compensated post-gain
    set_vst3_param_strict(plug, "trash_stg1_pre", float(np.clip(12.0 + drive * 1.4, 8.0, 20.0)))
    set_vst3_param_strict(plug, "trash_stg1_drive", float(np.clip(drive, 0.0, 10.0)))
    set_vst3_param_strict(plug, "trash_stg1_style", float(np.clip(style, 0.0, 1.0)))
    set_vst3_param_strict(plug, "trash_stg1_tension_scale", float(np.clip(32.0 + drive * 5.0, 20.0, 75.0)))
    set_vst3_param_strict(plug, "trash_stg1_gain", -6.0)
    set_vst3_param_strict(plug, "trash_stg1_mix", 100.0)
    set_vst3_param_strict(plug, "trash_stg1_filter_enable", True)
    set_vst3_param_strict(plug, "trash_stg1_filter_low_freq", float(np.clip(low_cut_hz, 120.0, 1200.0)))
    set_vst3_param_strict(plug, "trash_stg1_filter_high_freq", float(np.clip(high_cut_hz, 2000.0, 6500.0)))
    if "gain_output_lock" in plug.parameters:
        set_vst3_param_strict(plug, "gain_output_lock", True)
        set_vst3_param_strict(plug, "gain_output_locked_db", -3.0)
    set_vst3_param_strict(plug, "global_mix", float(np.clip(wet_mix, 0.0, 100.0)))
    set_vst3_param_strict(plug, "limiter_enable", True)
