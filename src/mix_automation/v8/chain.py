"""v8 R3: processing skeleton on the real VST3 plugins (lead, sends, backing track, 2mix bus).

Contract: a plugin that fails to load or to accept a parameter raises (no silent pass-through, no
substitute). Every stage is logged with plugin name, path, loaded flag, bypass flag, the parameters
that differ from the plugin's load-time state, and what was measured (GR / attenuation).

Sources: rulebook `ID@m:ss` = docs/mix_knowledge/rulebook.md. "R3 既定" = R3 default inside the
rulebook range, to be tuned in R4/R5 via knobs.py.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from mix_automation.config.plugin_registry import VST3_ROOT, PluginRole, PluginSpec, get_plugin_map
from mix_automation.dsp.vst3_adapter import (
    configure_early_room_phoenix,
    configure_rx10_voice_denoise,
    configure_softube_saturation,
    configure_softube_vca,
    configure_uad_la2a,
    configure_uad_pultec,
    set_vst3_param_strict,
)
from mix_automation.v8 import correction_eq as ceq
from mix_automation.v8 import dsp
from mix_automation.v8.knobs import DELAY_TYPES, R3_DELAY_TYPE, V8Knobs, pultec_air_kcs
from mix_automation.v8.metrics._core import band_power, stft_power

# kHs Limiter is not in the v7 catalog (adding it there could change v7 role lookups), so it is
# described here. Same loadable_path rule as every other PluginSpec.
KHS_LIMITER = PluginSpec(
    key="khs_limiter",
    name="Kilohearts kHs Limiter",
    role=PluginRole.TRUE_PEAK_LIMITER,
    path=VST3_ROOT / "Kilohearts" / "kHs Limiter.vst3",
)

# ---- working levels
# Lead normalised to an active (P.56) level of -18 dBFS: 0VU = -18 dBFS convention (-c0bHeO5ISs@5:30).
LEAD_WORK_LEVEL_DB = -18.0
# Backing track to -18 LUFS: mastered off-vocal (~-9 LUFS) pulled down 8-9 dB (RTVPdDmWrwo@20:10, E8zUKz21S9c@1:37).
INST_WORK_LUFS = -18.0
# 2mix bus entered at -14 LUFS, then raised to the landing level (RTVPdDmWrwo@22:34).
MIX_WORK_LUFS = -14.0
# Landing loudness -10..-9 LUFS (RTVPdDmWrwo@20:10); aim at the middle.
MASTER_TARGET_LUFS = -9.5
MASTER_LUFS_RANGE = (-10.0, -9.0)


@dataclass
class StageLog:
    stages: list[dict] = field(default_factory=list)

    def add(self, rec: dict) -> dict:
        self.stages.append(rec)
        return rec


# ------------------------------------------------------------------ plugin helpers


_PLUGIN_CACHE: dict[str, Any] | None = None
# Reloaded every time even inside plugin_cache(): a reused PhoenixVerb instance rejected the second
# rvb_time write (wrote 0.448, read 0.084; R4 smoke test 2026-11-27).
# iZotope Vocal Doubler too: a reused instance gave a different output on every call (same settings,
# second call +2.2 dB apart), while fresh loads were bit-identical across processes
# (scripts/match/parallel_safety.py, data/processed/v8r_match/parallel_safety/20261128-032827.json).
# R5 correction EQ: same Persistent Q binary as the subtractive EQ under its own key, never reused: the
# correction file may change global parameters (output gain, latency mode) that lead_tone does not reset.
PERSISTENT_Q_CORRECTION_KEY = "persistent_q_correction"
_NO_CACHE_KEYS = frozenset({"phoenix_verb", "vocal_doubler", PERSISTENT_Q_CORRECTION_KEY})

# Calibration warm starts (R4b): last calibrated value per stage key, used as x0 for the next calibration.
# Only active inside plugin_cache() (matching); None outside, so engine renders calibrate from the ends.
_CALIB_HINTS: dict[str, float] | None = None
# Warm-start step per stage (config, about 1/10 of each calibration range: VCA -40..-8 dB, LA-2A peak
# reduction 10..90, De-ess threshold -50..-2 dB, kHs threshold window peak-9..peak-0.5 dB) and whether
# the measured reduction grows with the parameter (matches the lo/hi passed to dsp.calibrate below).
_CALIB_STEP = {"vca": 3.0, "la2a": 10.0, "deess": 4.0, "limiter": 1.5}
_CALIB_INCREASING = {"vca": False, "la2a": True, "deess": False, "limiter": False}


@contextlib.contextmanager
def plugin_cache():
    """Reuse one instance per plugin key inside the block (R4 matching runs the chain many times).

    Every stage sets all parameters it relies on and run_plugin resets the plugin state, so a reused
    instance renders the same as a fresh one. Not used by the engine (one render per load).
    Also enables calibration warm starts (_CALIB_HINTS) for the block."""
    global _PLUGIN_CACHE, _CALIB_HINTS
    prev, prev_h = _PLUGIN_CACHE, _CALIB_HINTS
    _PLUGIN_CACHE = {} if prev is None else prev
    _CALIB_HINTS = {} if prev_h is None else prev_h
    try:
        yield
    finally:
        _PLUGIN_CACHE, _CALIB_HINTS = prev, prev_h


def set_calib_hints(hints: dict[str, float] | None) -> dict[str, float] | None:
    """Swap the warm-start store (match_pool keeps one per song); returns the previous one."""
    global _CALIB_HINTS
    prev = _CALIB_HINTS
    _CALIB_HINTS = hints
    return prev


def _calibrate(key: str, render, measure, target: float, lo: float, hi: float):
    """dsp.calibrate with the stage's warm start when hints are active; stores the result."""
    hints = _CALIB_HINTS
    if hints is None:
        return dsp.calibrate(render, measure, target, lo, hi)
    cal = dsp.calibrate(render, measure, target, lo, hi, x0=hints.get(key), step=_CALIB_STEP[key],
                        increasing=_CALIB_INCREASING[key])
    hints[key] = float(cal.value)
    return cal


def load_stage_plugin(spec: PluginSpec) -> tuple[Any, dict]:
    """Load `spec` from its loadable path (inner x86_64-win binary for bundles). Raises on failure."""
    from pedalboard import load_plugin

    path = spec.loadable_path
    if not path.exists():
        raise FileNotFoundError(f"v8: plugin binary not found for {spec.name}: {path}")
    use_cache = _PLUGIN_CACHE is not None and spec.key not in _NO_CACHE_KEYS
    if use_cache and spec.key in _PLUGIN_CACHE:
        plug = _PLUGIN_CACHE[spec.key]
    else:
        plug = load_plugin(str(path))
        if use_cache:
            _PLUGIN_CACHE[spec.key] = plug
    return plug, {"plugin": spec.name, "key": spec.key, "path": str(path), "loaded": True, "bypassed": False}


def _snapshot(plug: Any) -> dict[str, str]:
    snap = {}
    for k in plug.parameters:
        with contextlib.suppress(Exception):  # a non-readable parameter is simply not logged
            snap[k] = str(getattr(plug, k))
    return snap


def _changed(before: dict[str, str], plug: Any) -> dict[str, str]:
    after = _snapshot(plug)
    return {k: v for k, v in after.items() if before.get(k) != v}


def fit_len(y: np.ndarray, n: int) -> np.ndarray:
    if y.shape[-1] == n:
        return y
    if y.shape[-1] > n:
        return y[..., :n]
    pad = [(0, 0)] * (y.ndim - 1) + [(0, n - y.shape[-1])]
    return np.pad(y, pad)


def run_plugin(plug: Any, x: np.ndarray, sr: int) -> np.ndarray:
    """Mono in -> duplicated to 2 ch -> averaged back; stereo in -> stereo out. Same length as input."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        y = plug(np.stack([a, a]), sr, reset=True)
        y = y.mean(axis=0) if y.ndim == 2 else y
    else:
        y = plug(a, sr, reset=True)
    return fit_len(np.asarray(y, dtype=np.float32), a.shape[-1])


_NUM = re.compile(r"\d+(?:\.\d+)?")


def set_khs_db(plug: Any, key: str, value_db: float, lo_db: float, hi_db: float) -> str:
    """kHs dB knobs expose display strings with a mis-encoded minus sign, so write the normalised raw
    value (linear in dB, verified against the -20..+6 dB threshold list) and verify by reading back."""
    if not lo_db <= value_db <= hi_db:
        raise ValueError(f"{key}={value_db} dB outside [{lo_db}, {hi_db}]")
    param = plug.parameters[key]
    param.raw_value = (value_db - lo_db) / (hi_db - lo_db)
    shown = str(getattr(plug, key))
    m = _NUM.search(shown)
    if not m:
        raise RuntimeError(f"kHs read-back of {key!r} not numeric: {shown!r}")
    mag = float(m.group(0))
    val = mag if shown.strip().startswith("+") or mag == 0.0 else -mag
    if abs(val - value_db) > 0.06:
        raise RuntimeError(f"kHs write verification failed on {key!r}: wrote {value_db}, read {shown!r}")
    return shown


def _static_gain_comp_gr(before: np.ndarray, after: np.ndarray, neutral: np.ndarray, mask: np.ndarray, sr: int) -> float:
    """Compressor GR with make-up removed: level drop vs. the same plugin at a no-compression setting."""
    return dsp.gr_db(before, after, mask, sr) - dsp.gr_db(before, neutral, mask, sr)


def _level_to(x: np.ndarray, sr: int, target_db: float) -> tuple[np.ndarray, float]:
    act, _ = dsp.p56_threshold_db(x, sr)
    g = float(target_db - act)
    return (x * 10.0 ** (g / 20.0)).astype(np.float32), g


# ------------------------------------------------------------------ lead chain


def process_lead(lead: np.ndarray, inst_work: np.ndarray, sr: int, knobs: V8Knobs, log: StageLog) -> tuple[np.ndarray, dict]:
    x, mask = lead_front(lead, sr, log)
    x = lead_tone(x, mask, sr, knobs, log)
    return lead_ride(x, inst_work, sr, knobs, log)


def lead_front(lead: np.ndarray, sr: int, log: StageLog) -> tuple[np.ndarray, np.ndarray]:
    """Input trim + RX De-click + RX Voice De-noise: no knob reaches these, so R4 caches the result.

    Returns (audio, P.56 active mask of the trimmed input); the mask is reused by lead_tone as in R3."""
    pmap = get_plugin_map()
    x, g0 = _level_to(lead, sr, LEAD_WORK_LEVEL_DB)
    log.add({"stage": "lead_input_trim", "plugin": None, "gain_db": round(g0, 3), "target_active_dbfs": LEAD_WORK_LEVEL_DB})
    mask = dsp.active_mask(x, sr)

    # 1a. RX 10 Mouth De-click, sensitivity = plugin default 4.0 (RTVPdDmWrwo@4:56 uses De-click at 6; R3 keeps it lighter).
    plug, rec = load_stage_plugin(pmap["rx10_mouth_declick"])
    before = _snapshot(plug)
    set_vst3_param_strict(plug, "global_bypass", False)
    set_vst3_param_strict(plug, "output_clicks_only", False)
    set_vst3_param_strict(plug, "sensitivity", 4.0)
    rec["params"] = _changed(before, plug) | {"sensitivity": str(plug.sensitivity)}
    y = run_plugin(plug, x, sr)
    rec["measured"] = {"level_change_db": round(-dsp.gr_db(x, y, mask, sr), 3)}
    log.add({"stage": "lead_mouth_declick", **rec})
    x = y

    # 1b. RX 10 Voice De-noise, reduction 4.2 dB (existing profile configure_rx10_voice_denoise; "最低限" RTVPdDmWrwo@5:56).
    plug, rec = load_stage_plugin(pmap["rx10_voice_denoise"])
    before = _snapshot(plug)
    configure_rx10_voice_denoise(plug, reduction_db=4.2)
    rec["params"] = _changed(before, plug)
    y = run_plugin(plug, x, sr)
    rec["measured"] = {"level_change_db": round(-dsp.gr_db(x, y, mask, sr), 3)}
    log.add({"stage": "lead_voice_denoise", **rec})
    return y, mask


def lead_tone(x: np.ndarray, mask: np.ndarray, sr: int, knobs: V8Knobs, log: StageLog) -> np.ndarray:
    """EQ, pre-levelling, compressors (+ parallel), Pultec, de-ess, saturation, limiter."""
    pmap = get_plugin_map()

    # 2. Persistent Q: HPF = hpf_hz (R3: 140 Hz between 119 Hz Mo_qfyNdYQQ@2:38 and 159 Hz 2afvdjOpVJQ@2:40),
    #    18 dB/oct (existing profile configure_persistent_q_5band); 500 Hz -1.5 dB Q 1.0 ("少し"
    #    2afvdjOpVJQ@4:17, R3 既定); band 3 presence peak (articles [3][8]) only when presence_db > 0.
    plug, rec = load_stage_plugin(pmap["persistent_q"])
    before = _snapshot(plug)
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "mix", 100.0)
    bands = {1: ("Highpass", float(knobs.hpf_hz), 0.0, 0.7), 2: ("Peaking", 500.0, -1.5, 1.0)}
    if knobs.presence_db > 0.0:
        # Q 1.0: same width as the R3 500 Hz band (about 1.4 octaves).
        bands[3] = ("Peaking", float(knobs.presence_hz), float(knobs.presence_db), 1.0)
    for b, (typ, freq, gain, q) in bands.items():
        set_vst3_param_strict(plug, f"band_{b}_enabled", True)
        set_vst3_param_strict(plug, f"band_{b}_type", typ)
        set_vst3_param_strict(plug, f"band_{b}_placement", "Stereo")
        set_vst3_param_strict(plug, f"band_{b}_frequency", freq)
        if typ == "Highpass":
            set_vst3_param_strict(plug, f"band_{b}_slope", 18.0)
        else:
            set_vst3_param_strict(plug, f"band_{b}_gain", gain)
            set_vst3_param_strict(plug, f"band_{b}_q", q)
        set_vst3_param_strict(plug, f"band_{b}_dynamic", False)
        set_vst3_param_strict(plug, f"band_{b}_spectral", False)
    b = len(bands) + 1
    while f"band_{b}_enabled" in plug.parameters:
        set_vst3_param_strict(plug, f"band_{b}_enabled", False)
        b += 1
    rec["params"] = _changed(before, plug)
    y = run_plugin(plug, x, sr)
    rec["measured"] = {"level_change_db": round(-dsp.gr_db(x, y, mask, sr), 3)}
    log.add({"stage": "lead_subtractive_eq", **rec})
    x = y

    # 2b. Pre-compressor levelling (clip gain, articles [2][8]); off at 0.
    if knobs.pre_level_strength > 0.0:
        x, st = dsp.pre_level(x, sr, knobs.pre_level_strength)
        log.add({"stage": "lead_pre_level", "plugin": None, "dsp": "v8.dsp.pre_level", **st})

    # 3. Softube VCA, fast: ratio 4, attack 0.4 (fast end), release 0.85 (plugin default, mid); target GR = 45 % of total.
    plug, rec = load_stage_plugin(pmap["vca_compressor"])
    before = _snapshot(plug)
    vca_kw = {"ratio": 4.0, "attack_norm": 0.4, "release_norm": 0.85, "sc_hpf_hz": 120.0, "output_db": 0.0, "dry_wet": 100.0}
    configure_softube_vca(plug, threshold_db=10.0, **vca_kw)
    neutral = run_plugin(plug, x, sr)  # threshold +10 dBFS: no gain reduction, only static gain

    def vca_render(thr: float) -> np.ndarray:
        set_vst3_param_strict(plug, "compression_db", float(thr))
        return run_plugin(plug, x, sr)

    target = knobs.comp_total_gr_db * 0.45
    cal = _calibrate("vca", vca_render, lambda y_: _static_gain_comp_gr(x, y_, neutral, mask, sr), target, -40.0, -8.0)
    set_vst3_param_strict(plug, "compression_db", float(cal.value))
    rec["params"] = _changed(before, plug)
    rec["calibration"] = cal.summary()
    rec["measured"] = {"gr_db": round(cal.measured, 3), "target_gr_db": round(target, 3)}
    log.add({"stage": "lead_comp1_vca", **rec})
    x = cal.audio

    # 4. UAD LA-2A, Comp mode, gain 35 (existing profile configure_uad_la2a); target GR = 55 % of total.
    plug, rec = load_stage_plugin(pmap["uad_la2a"])
    before = _snapshot(plug)
    configure_uad_la2a(plug, peak_reduct=0.0, gain=35.0, mode="Comp")
    neutral = run_plugin(plug, x, sr)

    def la2a_render(pr: float) -> np.ndarray:
        set_vst3_param_strict(plug, "peak_reduct", round(pr))
        return run_plugin(plug, x, sr)

    target = knobs.comp_total_gr_db * 0.55
    cal = _calibrate("la2a", la2a_render, lambda y_: _static_gain_comp_gr(x, y_, neutral, mask, sr), target, 10.0, 90.0)
    set_vst3_param_strict(plug, "peak_reduct", round(cal.value))
    rec["params"] = _changed(before, plug)
    rec["calibration"] = cal.summary()
    rec["measured"] = {"gr_db": round(cal.measured, 3), "target_gr_db": round(target, 3)}
    log.add({"stage": "lead_comp2_la2a", **rec})
    x = cal.audio
    # 4b. Parallel compression (articles [8]: squashed copy, GR 10-20 dB); off at 0.
    if knobs.par_comp_mix > 0.0:
        x, st = dsp.parallel_compress(x, sr, knobs.par_comp_mix)
        log.add({"stage": "lead_parallel_comp", "plugin": None, "dsp": "v8.dsp.parallel_compress", **st})
    x, g1 = _level_to(x, sr, LEAD_WORK_LEVEL_DB)
    log.add({"stage": "lead_post_comp_trim", "plugin": None, "gain_db": round(g1, 3), "target_active_dbfs": LEAD_WORK_LEVEL_DB})

    # 5. UAD Pultec EQP-1A: 100 CPS boost 2 / atten 3 (-c0bHeO5ISs@19:22), hf boost = brightness at
    #    air_hz rounded to 8/10/12 KCS (R3: 8 KCS), hf_q 4.5 (existing profile).
    hf_freq, air_used = pultec_air_kcs(knobs.air_hz)
    plug, rec = load_stage_plugin(pmap["uad_pultec_eqp1a"])
    before = _snapshot(plug)
    configure_uad_pultec(plug, lf_boost=2.0, hf_boost=knobs.brightness, lf_freq="100 CPS", hf_freq=hf_freq, hf_q=4.5)
    set_vst3_param_strict(plug, "lf_atten", 3.0)
    rec["params"] = _changed(before, plug)
    rec["air_hz_rounding"] = {"knob_hz": float(knobs.air_hz), "used_hz": air_used, "hf_freq": hf_freq}
    y = run_plugin(plug, x, sr)
    rec["measured"] = {"level_change_db": round(-dsp.gr_db(x, y, mask, sr), 3)}
    log.add({"stage": "lead_tone_eq_pultec", **rec})
    x = y

    # 6. RX 10 De-ess (spectral), one stage; threshold calibrated so sibilant frames lose `deess_db` in 5-10 kHz.
    x = _deess_stage(x, sr, knobs, mask, log)

    # 7. Saturation Knob: NEUTRAL type, amount = saturation (OFF at 0) (2afvdjOpVJQ@8:02).
    plug, rec = load_stage_plugin(pmap["saturation_knob"])
    before = _snapshot(plug)
    configure_softube_saturation(plug, amount=knobs.saturation, sat_type="NEUTRAL")
    rec["params"] = _changed(before, plug) | {"saturation": str(plug.saturation)}
    y = run_plugin(plug, x, sr)
    rec["measured"] = {"level_change_db": round(-dsp.gr_db(x, y, mask, sr), 3)}
    log.add({"stage": "lead_saturation", **rec})
    x = y

    # 8. kHs Limiter last in the lead chain, peak GR about 4 dB (RTVPdDmWrwo@21:41); target 3.5 = middle of 3-4.
    return _limiter_stage(x, sr, log)


def lead_ride(x: np.ndarray, inst_work: np.ndarray, sr: int, knobs: V8Knobs, log: StageLog) -> tuple[np.ndarray, dict]:
    # 9. Phrase ride against the backing track.
    ride = dsp.phrase_ride(x, inst_work, sr, target_lu=knobs.vocal_offset_lu, tol_lu=knobs.ride_tolerance_lu, max_db=6.0)
    st = ride.stats()
    log.add({"stage": "lead_phrase_ride", "plugin": None, "target_lu": knobs.vocal_offset_lu, "tol_lu": knobs.ride_tolerance_lu,
             **{k: v for k, v in st.items() if k != "phrases"}})
    return ride.audio, st


_DEESS_BAND = (5000.0, 10000.0)
_SIBILANT_TOP_SHARE = 0.10  # top 10 % of active frames by 5-10 kHz share (R3 既定)


def sibilant_mask(x: np.ndarray, sr: int, active: np.ndarray) -> np.ndarray:
    freqs, _, p = stft_power(x, sr)
    share = band_power(p, freqs, *_DEESS_BAND) / (p.sum(axis=0) + 1e-20)
    act = np.asarray(active[: len(share)], dtype=bool)
    if not np.any(act):
        return np.zeros(len(share), dtype=bool)
    thr = np.quantile(share[act], 1.0 - _SIBILANT_TOP_SHARE)
    return act & (share >= thr)


def band_attenuation_db(before: np.ndarray, after: np.ndarray, sr: int, frames: np.ndarray, band: tuple[float, float]) -> float:
    freqs, _, pb = stft_power(before, sr)
    _, _, pa = stft_power(after, sr)
    bb = band_power(pb, freqs, *band)
    ba = band_power(pa, freqs, *band)
    m = frames[: len(bb)]
    if not np.any(m):
        return 0.0
    return float(np.mean(10 * np.log10(bb[m] + 1e-20) - 10 * np.log10(ba[m] + 1e-20)))


def _deess_stage(x: np.ndarray, sr: int, knobs: V8Knobs, mask: np.ndarray, log: StageLog) -> np.ndarray:
    plug, rec = load_stage_plugin(get_plugin_map()["rx10_deess"])
    if knobs.deess_db <= 0.0:
        rec.update({"bypassed": True, "reason": "knob deess_db=0", "params": {}})
        log.add({"stage": "lead_deess", **rec})
        return x
    before = _snapshot(plug)
    # Spectral De-ess like configure_rx10_spectral_deess, cutoff 5.2 kHz (its lowest allowed, closest to the 5 kHz band edge);
    # threshold is set directly over the full -60..0 dB range because the calibrated value may be below -18.
    set_vst3_param_strict(plug, "global_bypass", False)
    set_vst3_param_strict(plug, "output_ess_only", False)
    set_vst3_param_strict(plug, "algorithm", "Spectral De-ess")
    set_vst3_param_strict(plug, "absolute_mode", False)
    set_vst3_param_strict(plug, "cutoff_freq", 5200.0)
    set_vst3_param_strict(plug, "spectral_shaping", 45.0)
    set_vst3_param_strict(plug, "speed", "Fast")
    sib = sibilant_mask(x, sr, mask)

    def render(thr: float) -> np.ndarray:
        set_vst3_param_strict(plug, "threshold", float(thr))
        return run_plugin(plug, x, sr)

    cal = _calibrate("deess", render, lambda y_: band_attenuation_db(x, y_, sr, sib, _DEESS_BAND), knobs.deess_db, -50.0, -2.0)
    set_vst3_param_strict(plug, "threshold", float(cal.value))
    rec["params"] = _changed(before, plug)
    rec["calibration"] = cal.summary()
    rec["measured"] = {
        "sibilant_5_10k_atten_db": round(cal.measured, 3),
        "target_db": knobs.deess_db,
        "other_active_5_10k_atten_db": round(band_attenuation_db(x, cal.audio, sr, mask[: len(sib)] & ~sib, _DEESS_BAND), 3),
        "n_sibilant_frames": int(np.sum(sib)),
    }
    log.add({"stage": "lead_deess", **rec})
    return cal.audio


_LIMITER_TARGET_PEAK_GR_DB = 3.5  # RTVPdDmWrwo@21:41 "GR 約 4dB"; R3 aims at 3.5 (middle of 3-4)
_LIM_BLOCK_S = 0.01


def limiter_peak_gr_db(before: np.ndarray, after: np.ndarray, sr: int, thr_db: float) -> float:
    """99th percentile of 10 ms block peak reduction, with the limiter's own make-up gain removed
    (estimated from blocks 6 dB or more below the threshold, where the limiter is linear)."""
    blk = max(1, round(_LIM_BLOCK_S * sr))
    n = (min(len(before), len(after)) // blk) * blk
    pb = np.max(np.abs(before[:n].reshape(-1, blk)), axis=1)
    pa = np.max(np.abs(after[:n].reshape(-1, blk)), axis=1)
    db_b = 20 * np.log10(pb + 1e-12)
    db_a = 20 * np.log10(pa + 1e-12)
    loud = db_b > db_b.max() - 40.0
    lin = loud & (db_b < thr_db - 6.0)
    offset = float(np.median(db_a[lin] - db_b[lin])) if np.any(lin) else 0.0
    red = (db_b - db_a) + offset
    return float(np.percentile(red[loud], 99)) if np.any(loud) else 0.0


def _limiter_stage(x: np.ndarray, sr: int, log: StageLog) -> np.ndarray:
    plug, rec = load_stage_plugin(KHS_LIMITER)
    before = _snapshot(plug)
    set_vst3_param_strict(plug, "bypass", False)
    peak_db = float(20 * np.log10(np.max(np.abs(x)) + 1e-12))
    lo = float(np.clip(peak_db - 9.0, -20.0, 6.0))
    hi = float(np.clip(peak_db - 0.5, -20.0, 6.0))

    def render(thr: float) -> np.ndarray:
        set_khs_db(plug, "threshold", float(thr), -20.0, 6.0)
        return run_plugin(plug, x, sr)

    holder: dict[str, float] = {}

    def measure(y_: np.ndarray) -> float:
        thr = holder["thr"]
        return limiter_peak_gr_db(x, y_, sr, thr)

    def render_t(thr: float) -> np.ndarray:
        holder["thr"] = thr
        return render(thr)

    cal = _calibrate("limiter", render_t, measure, _LIMITER_TARGET_PEAK_GR_DB, hi, lo)
    set_khs_db(plug, "threshold", float(cal.value), -20.0, 6.0)
    rec["params"] = _changed(before, plug)
    rec["calibration"] = cal.summary()
    rec["measured"] = {"peak_gr_p99_db": round(cal.measured, 3), "target_peak_gr_db": _LIMITER_TARGET_PEAK_GR_DB,
                       "input_peak_dbfs": round(peak_db, 2)}
    log.add({"stage": "lead_limiter", **rec})
    # Output trim back to the working level so the ride starts from a known place (digital gain only).
    y, g = _level_to(cal.audio, sr, LEAD_WORK_LEVEL_DB)
    log.add({"stage": "lead_post_limiter_trim", "plugin": None, "gain_db": round(g, 3)})
    return y


# ------------------------------------------------------------------ sends


MIN_DELAY_S = 0.1  # R3: below ~100 ms a tap fuses with the voice like a double; rulebook delays are >=150 ms or 1/16


def delay_note_s(tempo_bpm: float, start: str = "1/16") -> tuple[float, str]:
    """1/16 ping-pong (2afvdjOpVJQ@15:50); doubled to 1/8 when 1/16 is shorter than MIN_DELAY_S.

    Beat This! can report double tempo (シックス: 187.5 on the off-vocal vs 93.8 on the master), so the
    note value is chosen by time, not only by the counted beat. `start` = shortest note allowed.
    """
    beat = 60.0 / tempo_bpm
    notes = (("1/16", 0.25), ("1/8", 0.5), ("1/4", 1.0))
    first = [nm for nm, _ in notes].index(start)
    for name, frac in notes[first:]:
        if beat * frac >= MIN_DELAY_S:
            return beat * frac, name
    return beat, "1/4"


SLAP_DELAY_S = 0.12  # articles [8] slap 80-120 ms; fixed at the upper end (not moved by the optimiser)


def render_delay(lead_dry: np.ndarray, sr: int, delay_type: int, tempo_bpm: float) -> tuple[np.ndarray, dict]:
    """Delay return at unity send (stereo). delay_type: see knobs.DELAY_TYPES; -1 = R3 ping-pong."""
    if delay_type == R3_DELAY_TYPE:
        # R3: 1/16 ping-pong (2afvdjOpVJQ@15:50); HP 400 / LP 4 kHz inside the feedback.
        delay_s, note = delay_note_s(tempo_bpm)
        kw = {"feedback": 0.3, "hp": 400.0, "lp": 4000.0}
        y = dsp.pingpong_delay(lead_dry, sr, delay_s, **kw)
        name, desc = "v8.dsp.pingpong_delay", "R3 1/16 ping-pong"
    elif delay_type == 0:
        # articles [1]: 1/16 once, low; centred; R3 band HP 400 / LP 4 kHz (2afvdjOpVJQ@15:50).
        delay_s, note = delay_note_s(tempo_bpm)
        kw = {"feedback": 0.0, "hp": 400.0, "lp": 4000.0}
        y = dsp.feedback_delay(lead_dry, sr, delay_s, n_taps=1, **kw)
        name, desc = "v8.dsp.feedback_delay", DELAY_TYPES[0]
    elif delay_type == 1:
        # articles [4]: 1/8, feedback 50 %, HPF 200 Hz, LPF 10 kHz; ping-pong placement as R3.
        delay_s, note = delay_note_s(tempo_bpm, start="1/8")
        kw = {"feedback": 0.5, "hp": 200.0, "lp": 10000.0}
        y = dsp.pingpong_delay(lead_dry, sr, delay_s, **kw)
        name, desc = "v8.dsp.pingpong_delay", DELAY_TYPES[1]
    elif delay_type == 2:
        # articles [8]: slap 80-120 ms, one tap, centred; R3 band HP 400 / LP 4 kHz.
        delay_s, note = SLAP_DELAY_S, "slap"
        kw = {"feedback": 0.0, "hp": 400.0, "lp": 4000.0}
        y = dsp.feedback_delay(lead_dry, sr, delay_s, n_taps=1, **kw)
        name, desc = "v8.dsp.feedback_delay", DELAY_TYPES[2]
    else:
        raise ValueError(f"unknown delay_type {delay_type}")
    rec = {"dsp": name, "delay_type": delay_type, "type": desc, "delay_s": round(delay_s, 4), "note": note,
           "feedback": kw["feedback"], "hp_hz": kw["hp"], "lp_hz": kw["lp"]}
    return y, rec


def build_sends(lead_dry: np.ndarray, sr: int, knobs: V8Knobs, tempo_bpm: float, log: StageLog) -> dict[str, np.ndarray]:
    pmap = get_plugin_map()
    sends: dict[str, np.ndarray] = {}
    stereo_in = np.stack([lead_dry, lead_dry]).astype(np.float32)

    # Doubler: effect only, amount/separation 100 ("D-Wider 100%" JUXJShxfzKQ@7:07) -> Mid -6 dB, LPF 9 kHz.
    plug, rec = load_stage_plugin(pmap["vocal_doubler"])
    before = _snapshot(plug)
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "effect_only", True)
    set_vst3_param_strict(plug, "amount", 100.0)
    set_vst3_param_strict(plug, "separation", 100.0)
    rec["params"] = _changed(before, plug) | {"amount": str(plug.amount), "separation": str(plug.separation)}
    wet = dsp.doubler_post(run_plugin(plug, stereo_in, sr), sr, center_db=-6.0, lpf_hz=9000.0)
    sends["doubler"] = (wet * 10 ** (knobs.doubler_send_db / 20)).astype(np.float32)
    rec["post"] = {"center_db": -6.0, "lpf_hz": 9000.0, "send_db": knobs.doubler_send_db}
    log.add({"stage": "send_doubler", **rec})

    # Room: PhoenixVerb small (0.45 s, pre-delay 10 ms, 6 m) -> HPF 100 Hz, LPF 2.7 kHz (JUXJShxfzKQ@8:49).
    plug, rec = load_stage_plugin(pmap["phoenix_verb"])
    before = _snapshot(plug)
    set_vst3_param_strict(plug, "bypass", False)
    configure_early_room_phoenix(plug, size_m=6.0, rvb_time=0.45, predelay_ms=10.0)
    rec["params"] = _changed(before, plug)
    wet = run_plugin(plug, stereo_in, sr)
    wet = dsp.butter_filter(dsp.butter_filter(wet, sr, "highpass", 100.0), sr, "lowpass", 2700.0)
    sends["room"] = (wet * 10 ** (knobs.room_send_db / 20)).astype(np.float32)
    rec["post"] = {"hpf_hz": 100.0, "lpf_hz": 2700.0, "send_db": knobs.room_send_db}
    log.add({"stage": "send_room", **rec})

    # Plate: Rev PLATE-140, pre-delay 50 ms (40-60), decay = knob, HP 400 Hz, 100 % wet -> LPF 3.4 kHz (JUXJShxfzKQ@14:30).
    # The plugin's post-EQ (default low shelf -24 dB @ 48 Hz, high shelf -12 dB @ 6 kHz) is switched off; the
    # rulebook high cut is applied as a scipy low-pass instead, since the plugin has no low-pass.
    plug, rec = load_stage_plugin(pmap["rev_plate_140"])
    before = _snapshot(plug)
    set_vst3_param_strict(plug, "on_off", "Active")
    set_vst3_param_strict(plug, "blend_dry_wet", 1.0)
    set_vst3_param_strict(plug, "pre_delay_ms", 50.0)
    set_vst3_param_strict(plug, "decay_time", float(knobs.plate_decay_s))
    set_vst3_param_strict(plug, "hp_filter_hz", 400.0)
    set_vst3_param_strict(plug, "post_eq_on_off", False)
    rec["params"] = _changed(before, plug)
    wet = dsp.butter_filter(run_plugin(plug, stereo_in, sr), sr, "lowpass", 3400.0)
    plate = wet * 10 ** (knobs.plate_send_db / 20)
    rec["post"] = {"lpf_hz": 3400.0, "send_db": knobs.plate_send_db}
    log.add({"stage": "send_plate", **rec})

    delay, drec = render_delay(lead_dry, sr, int(knobs.delay_type), tempo_bpm)
    delay = delay * 10 ** (knobs.delay_send_db / 20)
    log.add({"stage": "send_delay", "plugin": None, **drec, "tempo_bpm": round(tempo_bpm, 2),
             "send_db": knobs.delay_send_db})

    # Duck plate + delay returns while the lead sings (JUXJShxfzKQ@37:34). Key range = P.56 margin: 0 dB at the
    # activity threshold, full depth at the active level.
    act, thr = dsp.p56_threshold_db(lead_dry, sr)
    g = dsp.duck_gain(lead_dry, sr, knobs.fx_duck_db, thr, act - thr, attack_ms=20.0, release_ms=250.0)
    g = g[: plate.shape[-1]].astype(np.float32)
    sends["plate"] = (plate * g).astype(np.float32)
    sends["delay"] = (delay * g).astype(np.float32)
    gr = -20 * np.log10(g + 1e-12)
    log.add({"stage": "send_fx_duck", "plugin": None, "dsp": "v8.dsp.duck_gain", "targets": ["plate", "delay"],
             "depth_db": knobs.fx_duck_db, "thr_db": round(thr, 2), "range_db": round(act - thr, 2),
             "measured": {"mean_gr_db": round(float(np.mean(gr)), 3), "max_gr_db": round(float(np.max(gr)), 3)}})
    return sends


# ------------------------------------------------------------------ R5 correction EQ (end of the lead)


def correction_eq(lead_dry: np.ndarray, sr: int, cfg: dict | None, log: StageLog) -> np.ndarray:
    """Persistent Q set by ear (v8/correction_eq.py) on the lead after the phrase ride.

    Flat (no file / no band that changes the sound): returns `lead_dry` itself, no plugin is loaded and
    nothing is logged. Sends and the backing-track duck keep using the uncorrected lead."""
    if ceq.is_flat(cfg):
        return lead_dry
    spec = replace(get_plugin_map()["persistent_q"], key=PERSISTENT_Q_CORRECTION_KEY)
    plug, rec = load_stage_plugin(spec)
    before = _snapshot(plug)
    ceq.write_to_plugin(plug, cfg)
    rec["params"] = _changed(before, plug)
    y = run_plugin(plug, lead_dry, sr)
    rec["measured"] = {"lufs_change_db": round(float(dsp.integrated_lufs(y, sr) - dsp.integrated_lufs(lead_dry, sr)), 3)}
    rec["bands"] = ceq.active_bands(cfg)
    log.add({"stage": "lead_correction_eq", **rec})
    return y


def vocal_bus(lead: np.ndarray, sends: dict[str, np.ndarray], n: int) -> np.ndarray:
    """VocalBus before the 2mix bus: the lead on both channels plus every send (float64)."""
    bus = np.stack([lead, lead]).astype(np.float64)
    for w in sends.values():
        bus += fit_len(w, n)
    return bus


# ------------------------------------------------------------------ backing track


def prepare_inst(inst: np.ndarray, sr: int, log: StageLog) -> np.ndarray:
    li = dsp.integrated_lufs(inst, sr)
    if not np.isfinite(li):
        raise RuntimeError("v8: backing track is silent")
    g = INST_WORK_LUFS - li
    log.add({"stage": "inst_work_gain", "plugin": None, "input_lufs": round(li, 2), "gain_db": round(g, 3), "target_lufs": INST_WORK_LUFS})
    return (inst * 10 ** (g / 20)).astype(np.float32)


INST_SIDE_SHELF_HZ = 6000.0  # articles [2]: Side high end of the backing track ("6 kHz 以上", R4 指示)


def duck_inst(inst_work: np.ndarray, lead_dry: np.ndarray, sr: int, knobs: V8Knobs, log: StageLog) -> tuple[np.ndarray, dict]:
    out, st = dsp.multiband_duck(inst_work, lead_dry, sr, knobs.inst_duck_max_db)
    log.add({"stage": "inst_multiband_duck", "plugin": None, "dsp": "v8.dsp.multiband_duck", **st})
    if knobs.inst_side_high_db > 0.0:
        out = dsp.side_high_shelf(out, sr, knobs.inst_side_high_db, INST_SIDE_SHELF_HZ)
        log.add({"stage": "inst_side_high_shelf", "plugin": None, "dsp": "v8.dsp.side_high_shelf",
                 "gain_db": knobs.inst_side_high_db, "freq_hz": INST_SIDE_SHELF_HZ})
    return out, st


# ------------------------------------------------------------------ 2mix bus


def master_bus(mix: np.ndarray, sr: int, knobs: V8Knobs, log: StageLog) -> tuple[np.ndarray, dict]:
    pmap = get_plugin_map()
    lm = dsp.integrated_lufs(mix, sr)
    g_in = MIX_WORK_LUFS - lm
    x = (mix * 10 ** (g_in / 20)).astype(np.float32)
    log.add({"stage": "mix_work_gain", "plugin": None, "input_lufs": round(lm, 2), "gain_db": round(g_in, 3)})
    mask = dsp.active_mask(x, sr)

    # Saturation Knob light, KEEP LOW 0.8 (RTVPdDmWrwo@41:31-43:25, R3 既定).
    plug, rec = load_stage_plugin(pmap["saturation_knob"])
    before = _snapshot(plug)
    configure_softube_saturation(plug, amount=0.8, sat_type="KEEP LOW")
    rec["params"] = _changed(before, plug) | {"saturation": str(plug.saturation)}
    y = run_plugin(plug, x, sr)
    rec["measured"] = {"level_change_db": round(-dsp.gr_db(x, y, mask, sr), 3)}
    log.add({"stage": "mix_saturation", **rec})
    x = y

    # M/S compression on Mid, mean GR 1.5 dB (R3 既定), ratio 2, 10/100 ms.
    env_max = float(np.max(dsp.envelope(0.5 * (x[0] + x[1]), sr, 10.0, 100.0)))
    cal = dsp.calibrate(lambda t: dsp.ms_compress(x, sr, t), lambda r: r[1]["mean_gr_db"], 1.5, env_max - 30.0, env_max)
    x, ms_stats = cal.audio
    log.add({"stage": "mix_ms_compress", "plugin": None, "dsp": "v8.dsp.ms_compress", "calibration": cal.summary(), **ms_stats})

    # Parallel glue: Softube VCA ratio 3 (E8zUKz21S9c@14:17), attack 2.1 / release 0.7 / SC HPF 110 / drive 0.2
    # (existing v7 bus profile apply_2mix_bus_glue_and_safe_master). Wet-only GR calibrated to 4 dB (R3 既定), then dry/wet.
    plug, rec = load_stage_plugin(pmap["vca_compressor"])
    before = _snapshot(plug)
    glue_kw = {"ratio": 3.0, "attack_norm": 2.1, "release_norm": 0.7, "sc_hpf_hz": 110.0, "output_db": 0.0, "output_drive": 0.2}
    configure_softube_vca(plug, threshold_db=10.0, dry_wet=100.0, **glue_kw)
    neutral = run_plugin(plug, x, sr)

    def render(thr: float) -> np.ndarray:
        set_vst3_param_strict(plug, "compression_db", float(thr))
        return run_plugin(plug, x, sr)

    cal = dsp.calibrate(render, lambda y_: _static_gain_comp_gr(x, y_, neutral, mask, sr), 4.0, -32.0, -8.0)
    set_vst3_param_strict(plug, "compression_db", float(cal.value))
    set_vst3_param_strict(plug, "dry_wet_wet", float(knobs.glue_wet * 100.0))
    rec["params"] = _changed(before, plug)
    rec["calibration"] = cal.summary()
    y = run_plugin(plug, x, sr)
    rec["measured"] = {"wet_gr_db": round(cal.measured, 3), "target_wet_gr_db": 4.0,
                       "parallel_level_change_db": round(-dsp.gr_db(x, y, mask, sr) + dsp.gr_db(x, neutral, mask, sr), 3)}
    log.add({"stage": "mix_parallel_glue", **rec})
    glued = y

    # Wave Breaker limiter, threshold -0.8 dB (existing v7 master profile), pre-gain adjusted twice to land at -9.5 LUFS.
    plug, rec = load_stage_plugin(pmap["wave_breaker"])
    before = _snapshot(plug)
    set_vst3_param_strict(plug, "bypass", False)
    set_vst3_param_strict(plug, "threshold_db", -0.8)
    rec["params"] = _changed(before, plug) | {"threshold_db": str(plug.threshold_db), "normalize_db": str(plug.normalize_db)}
    lg = dsp.integrated_lufs(glued, sr)
    pre = MASTER_TARGET_LUFS - lg
    tries = []
    out = None
    lufs = float("nan")
    for _ in range(3):  # initial + 2 adjustments
        out = run_plugin(plug, (glued * 10 ** (pre / 20)).astype(np.float32), sr)
        lufs = dsp.integrated_lufs(out, sr)
        tries.append([round(pre, 3), round(lufs, 3)])
        if abs(lufs - MASTER_TARGET_LUFS) <= 0.1:
            break
        pre += MASTER_TARGET_LUFS - lufs
    assert out is not None
    if not MASTER_LUFS_RANGE[0] <= lufs <= MASTER_LUFS_RANGE[1]:
        raise RuntimeError(f"v8: master landed at {lufs:.2f} LUFS, outside {MASTER_LUFS_RANGE} after {tries}")
    tp = dsp.true_peak_dbfs(out)
    sp = float(20 * np.log10(np.max(np.abs(out)) + 1e-12))
    if sp > 0.0:
        raise RuntimeError(f"v8: master sample peak {sp:.2f} dBFS > 0")
    rec["measured"] = {"pre_gain_tries": tries, "lufs": round(lufs, 3), "true_peak_dbfs_approx": round(tp, 2), "sample_peak_dbfs": round(sp, 2)}
    log.add({"stage": "mix_limiter", **rec})
    stats = {
        "ms_gr_db": ms_stats["mean_gr_db"],
        "glue_wet_gr_db": round(cal.measured, 3),
        "master_lufs": round(lufs, 3),
        "true_peak_dbfs_approx": round(tp, 2),
        "input_lufs": round(lm, 3),
    }
    return out, stats
