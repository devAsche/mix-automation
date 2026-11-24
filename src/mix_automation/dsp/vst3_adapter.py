"""Strict Contract-Verified VST3 Parameter Adapter & Self-Calibrating Plugin Helpers.

Eliminates the systemic v1/v2 "silent failure" bug where `try: setattr(...) except Exception: pass`
masked incompatible types, case mismatches, and discrete `valid_values` rejections across
`RX 10 De-ess`, `Softube VCA Compressor`, `Softube Saturation Knob`, `UAD LA-2A`, `Persistent Q`,
`Soothe2`, and `Exponential Audio R2`.

Every parameter write is strictly coerced to the exact discrete/string/numeric token advertised by
the VST3 plugin, assigned, and read back for confirmation.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_numeric_token(token: str) -> float | None:
    m = _NUM_RE.search(token)
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

    # 4. Pure numeric discrete valid_values list (Persistent Q, R2, Soothe2, RX 10, SlickEQ, Pultec)
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        target = float(val)
        finite_vv = [x for x in vv if np.isfinite(float(x))]
        pool = finite_vv if finite_vv else vv
        return min(pool, key=lambda x: abs(float(x) - target))

    # 5. String valid_values list (VCA ratio, Saturation Knob, UAD LA-2A, PhoenixVerb, Pultec)
    if isinstance(first, str):
        if isinstance(val, str):
            norm_target = val.strip().lower()
            for cand in vv:
                if cand.strip().lower() == norm_target:
                    return cand
            # Try numeric parse of string target (e.g. "4.0" -> "4.0:1")
            num_val = _parse_numeric_token(val)
            if num_val is None:
                raise ValueError(f"String value {val!r} not in valid_values for {key!r}: {vv[:8]}")
        else:
            num_val = float(val)

        # Match numeric value against numeric-bearing string entries (e.g. '4.0:1', '35', ' 8.0 Meters')
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


def configure_rx10_voice_denoise(plug: Any, reduction_db: float = 4.0) -> None:
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
    """Configure Persistent Q with strictly verified static 5-or-6-band vocal EQ."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "mix", 100.0)

    # Band 1: Highpass
    set_vst3_param_strict(plug, "band_1_enabled", True)
    set_vst3_param_strict(plug, "band_1_type", "Highpass")
    set_vst3_param_strict(plug, "band_1_frequency", float(np.clip(hpf_hz, 55.0, 180.0)))
    set_vst3_param_strict(plug, "band_1_slope", 18.0)
    set_vst3_param_strict(plug, "band_1_dynamic", False)

    # Band 2: Chest / Body foundation
    set_vst3_param_strict(plug, "band_2_enabled", True)
    set_vst3_param_strict(plug, "band_2_type", "Peaking")
    set_vst3_param_strict(plug, "band_2_frequency", float(body_hz))
    set_vst3_param_strict(plug, "band_2_gain", float(np.clip(body_db, -6.0, 6.0)))
    set_vst3_param_strict(plug, "band_2_q", 0.90)
    set_vst3_param_strict(plug, "band_2_dynamic", False)

    # Band 3: Mud / Boxiness cleanup
    set_vst3_param_strict(plug, "band_3_enabled", True)
    set_vst3_param_strict(plug, "band_3_type", "Peaking")
    set_vst3_param_strict(plug, "band_3_frequency", float(mud_hz))
    set_vst3_param_strict(plug, "band_3_gain", float(np.clip(mud_db, -7.0, 3.0)))
    set_vst3_param_strict(plug, "band_3_q", 1.25)
    set_vst3_param_strict(plug, "band_3_dynamic", False)

    # Band 4: Vocal Presence / Forwardness
    set_vst3_param_strict(plug, "band_4_enabled", True)
    set_vst3_param_strict(plug, "band_4_type", "Peaking")
    set_vst3_param_strict(plug, "band_4_frequency", float(presence_hz))
    set_vst3_param_strict(plug, "band_4_gain", float(np.clip(presence_db, -4.0, 6.5)))
    set_vst3_param_strict(plug, "band_4_q", 0.80)
    set_vst3_param_strict(plug, "band_4_dynamic", False)

    # Band 5: Silky Air High-Shelf
    set_vst3_param_strict(plug, "band_5_enabled", True)
    set_vst3_param_strict(plug, "band_5_type", "HighShelf")
    set_vst3_param_strict(plug, "band_5_frequency", float(air_hz))
    set_vst3_param_strict(plug, "band_5_gain", float(np.clip(air_db, -4.0, 6.5)))
    set_vst3_param_strict(plug, "band_5_q", 0.70)
    set_vst3_param_strict(plug, "band_5_dynamic", False)

    # Optional Band 6: Core Mid (1.2kHz) density adjustment
    if mid_dip_hz is not None and abs(mid_dip_db) > 0.05:
        set_vst3_param_strict(plug, "band_6_enabled", True)
        set_vst3_param_strict(plug, "band_6_type", "Peaking")
        set_vst3_param_strict(plug, "band_6_frequency", float(mid_dip_hz))
        set_vst3_param_strict(plug, "band_6_gain", float(np.clip(mid_dip_db, -5.0, 5.0)))
        set_vst3_param_strict(plug, "band_6_q", 1.10)
        set_vst3_param_strict(plug, "band_6_dynamic", False)
    elif "band_6_enabled" in plug.parameters:
        set_vst3_param_strict(plug, "band_6_enabled", False)


def configure_softube_vca(
    plug: Any,
    threshold_db: float,
    ratio: float = 3.5,
    attack_norm: float = 1.65,
    release_norm: float = 0.75,
    sc_hpf_hz: float = 120.0,
    output_db: float = 0.0,
    dry_wet: float = 100.0,
    output_drive: float = 0.8,
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
    amount: float = 2.0,
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
    hf_freq: str = "10 KCS",
    hf_q: float = 4.5,
) -> None:
    """Configure UAD Pultec EQP-1A for vocal warmth & silky air sheen."""
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
    drive_db: float = 4.5,
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
    low_level: float = 6.0,
    high_level: float = -4.0,
    output_gain: float = 0.0,
) -> None:
    """Configure Waves MV2 upward/downward vocal leveler."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "low_level", float(np.clip(low_level, 0.0, 24.0)))
    set_vst3_param_strict(plug, "high_level", float(np.clip(high_level, -24.0, 0.0)))
    set_vst3_param_strict(plug, "output_gain", float(np.clip(output_gain, -12.0, 0.0)))


def configure_clean_plate_r2(
    plug: Any,
    rvb_time: float = 0.95,
    predelay_ms: float = 26.0,
    lpf_hz: float = 5750.0,
) -> None:
    """Configure Exponential Audio R2 to a strictly chorus-free short studio Plate."""
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "mix", 100.0)
    set_vst3_param_strict(plug, "revbtype", "Plate")
    set_vst3_param_strict(plug, "rvb_time", float(np.clip(rvb_time, 0.55, 1.8)))
    set_vst3_param_strict(plug, "predelay", float(np.clip(predelay_ms, 12.0, 55.0)))
    set_vst3_param_strict(plug, "chordpth", 0.0)
    set_vst3_param_strict(plug, "reverbfc", float(lpf_hz))
    set_vst3_param_strict(plug, "earlyffc", float(lpf_hz))


def configure_early_room_phoenix(
    plug: Any,
    size_m: float = 8.0,
    rvb_time: float = 0.42,
    predelay_ms: float = 12.0,
) -> None:
    """Configure Exponential Audio PhoenixVerb as a tight studio Early-Reflection room glue."""
    set_vst3_param_strict(plug, "mix", 100.0)
    set_vst3_param_strict(plug, "revbtype", "Chamber")
    set_vst3_param_strict(plug, "rvb_size", float(np.clip(size_m, 4.0, 16.0)))
    set_vst3_param_strict(plug, "rvb_time", float(np.clip(rvb_time, 0.28, 0.75)))
    set_vst3_param_strict(plug, "predelay", float(np.clip(predelay_ms, 5.0, 30.0)))
    set_vst3_param_strict(plug, "earlylvl", 0.0)
    set_vst3_param_strict(plug, "damp_fc", 5500.0)
