"""v8 R5: the correction EQ at the very end of the lead (Persistent Q set by ear, per singer).

The file profiles/correction_eq/<voice>.json holds the Persistent Q settings Ryo dialled in by ear
(part B reads them out of a Studio One .song with `bands_from_plugin`). Format "v8_correction_eq/1":

    {"format": "v8_correction_eq/1", "voice": "ryo", "plugin": "Persistent Q", "source": {...},
     "global": {<global parameters that differ from a freshly loaded Persistent Q>},
     "bands": [{"band": 1, "enabled": true, "type": "Peaking", "frequency_hz": 3000.0, "gain_db": -2.0,
                "q": 1.0, "slope_db_oct": null, "placement": "Stereo",
                "extra": {<other band parameters that differ from a fresh load>}}]}

Bands that are not in the file are disabled. A missing file, or one without any band that changes
the sound (see `is_flat`), means "flat": the stage is skipped and no plugin is loaded.

Parameter names, choices and ranges: read from Persistent Q through pedalboard on 2026-11-28
(32 bands; band_<n>_{enabled,type,placement,frequency,gain,q,slope,...}).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FORMAT = "v8_correction_eq/1"
PLUGIN = "Persistent Q"
ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "profiles" / "correction_eq"
DEFAULT_VOICE = "ryo"

# ---- Persistent Q parameter space (pedalboard plugin.parameters, valid_values / range)
N_BANDS = 32
BAND_TYPES = ("Peaking", "Highpass", "Lowpass", "HighShelf", "LowShelf", "Bandpass", "Notch", "Tilt")
PLACEMENTS = ("Stereo", "Left", "Right", "Mid", "Side")
SLOPES_DB_OCT = (6.0, 12.0, 18.0, 24.0)
FREQ_RANGE_HZ = (20.0, 22000.0)
GAIN_RANGE_DB = (-30.0, 30.0)
Q_RANGE = (0.1, 40.0)
PASS_TYPES = ("Highpass", "Lowpass")  # the types whose slope matters
# a band of these types with gain 0 (and no dynamic / spectral processing) leaves the signal unchanged
GAIN_TYPES = ("Peaking", "HighShelf", "LowShelf")
# core band fields: JSON key -> parameter suffix
CORE = {"enabled": "enabled", "type": "type", "frequency_hz": "frequency", "gain_db": "gain", "q": "q",
        "slope_db_oct": "slope", "placement": "placement"}
EXTRA_SUFFIXES = (
    "mute", "solo", "eq_brush_generated",
    "dynamic", "dynamic_sidechain", "threshold", "dynamic_range", "ratio", "knee", "attack", "release",
    "spectral", "spectral_sidechain", "spectral_threshold", "spectral_range", "spectral_density",
    "spectral_ratio", "spectral_knee", "spectral_attack", "spectral_release",
)
GLOBAL_CHOICES: dict[str, tuple] = {
    "bypass": (False, True),
    "latency_mode": ("Zero Latency", "Natural Phase", "Linear Phase"),
    "eq_character": ("Digital", "Analog"),
    "steering": (False, True),
    "steering_style": ("The Pro", "Bulldozer", "Golden Ears", "The Painter", "Surgeon", "Harmonic Surfer",
                       "Vocal Whisperer", "Pocket Guard", "Old Knob Guy", "Stage Ninja", "Newbie", "Lazy Genius"),
    "spectral_quality": ("High Quality", "Low Latency", "Very Low Latency"),
}
GLOBAL_RANGES: dict[str, tuple[float, float]] = {"mix": (0.0, 100.0), "input_gain": (-24.0, 24.0),
                                                 "output_gain": (-24.0, 24.0)}
GLOBAL_PARAMS = (*GLOBAL_CHOICES, *GLOBAL_RANGES)


class CorrectionEQError(ValueError):
    pass


def _num(v: Any, what: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CorrectionEQError(f"{what} must be a number, got {v!r}")
    f = float(v)
    if not (lo <= f <= hi):
        raise CorrectionEQError(f"{what}={f} is outside [{lo}, {hi}]")
    return f


def _choice(v: Any, what: str, choices: tuple) -> Any:
    if v not in choices or (isinstance(v, bool) != isinstance(choices[0], bool)):
        raise CorrectionEQError(f"{what}={v!r} is not one of {list(choices)}")
    return v


def validate(cfg: dict) -> dict:
    """Check a correction EQ dict and return a normalised copy. Raises CorrectionEQError."""
    if not isinstance(cfg, dict):
        raise CorrectionEQError("correction EQ must be a JSON object")
    if cfg.get("format") != FORMAT:
        raise CorrectionEQError(f"format must be {FORMAT!r}, got {cfg.get('format')!r}")
    if cfg.get("plugin", PLUGIN) != PLUGIN:
        raise CorrectionEQError(f"plugin must be {PLUGIN!r}, got {cfg.get('plugin')!r}")
    unknown = set(cfg) - {"format", "voice", "plugin", "source", "global", "bands"}
    if unknown:
        raise CorrectionEQError(f"unknown top-level key(s): {sorted(unknown)}")
    g_in = cfg.get("global") or {}
    if not isinstance(g_in, dict):
        raise CorrectionEQError("global must be an object")
    g = {}
    for k, v in g_in.items():
        if k in GLOBAL_CHOICES:
            g[k] = _choice(v, f"global.{k}", GLOBAL_CHOICES[k])
        elif k in GLOBAL_RANGES:
            g[k] = _num(v, f"global.{k}", *GLOBAL_RANGES[k])
        else:
            raise CorrectionEQError(f"unknown global parameter {k!r}")
    bands_in = cfg.get("bands") or []
    if not isinstance(bands_in, list):
        raise CorrectionEQError("bands must be a list")
    bands, seen = [], set()
    for i, b in enumerate(bands_in):
        if not isinstance(b, dict):
            raise CorrectionEQError(f"bands[{i}] must be an object")
        unknown = set(b) - {"band", *CORE, "extra"}
        if unknown:
            raise CorrectionEQError(f"bands[{i}]: unknown key(s) {sorted(unknown)}")
        n = b.get("band")
        if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= N_BANDS):
            raise CorrectionEQError(f"bands[{i}].band must be an integer 1..{N_BANDS}, got {n!r}")
        if n in seen:
            raise CorrectionEQError(f"band {n} appears twice")
        seen.add(n)
        enabled = b.get("enabled", True)
        if not isinstance(enabled, bool):
            raise CorrectionEQError(f"band {n}: enabled must be true/false")
        slope = b.get("slope_db_oct")
        extra = b.get("extra") or {}
        if not isinstance(extra, dict):
            raise CorrectionEQError(f"band {n}: extra must be an object")
        bad = sorted(set(extra) - set(EXTRA_SUFFIXES))
        if bad:
            raise CorrectionEQError(f"band {n}: unknown extra parameter(s) {bad}")
        bands.append({
            "band": n,
            "enabled": enabled,
            "type": _choice(b.get("type", "Peaking"), f"band {n} type", BAND_TYPES),
            "frequency_hz": _num(b.get("frequency_hz"), f"band {n} frequency_hz", *FREQ_RANGE_HZ),
            "gain_db": _num(b.get("gain_db", 0.0), f"band {n} gain_db", *GAIN_RANGE_DB),
            "q": _num(b.get("q", 0.7), f"band {n} q", *Q_RANGE),
            "slope_db_oct": None if slope is None else float(_choice(slope, f"band {n} slope_db_oct", SLOPES_DB_OCT)),
            "placement": _choice(b.get("placement", "Stereo"), f"band {n} placement", PLACEMENTS),
            "extra": dict(extra),
        })
    bands.sort(key=lambda d: d["band"])
    return {"format": FORMAT, "voice": cfg.get("voice"), "plugin": PLUGIN, "source": cfg.get("source") or {},
            "global": g, "bands": bands}


def _band_changes_sound(b: dict) -> bool:
    if not b["enabled"]:
        return False
    ex = b.get("extra") or {}
    if ex.get("dynamic") is True or ex.get("spectral") is True:
        return True
    return not (b["type"] in GAIN_TYPES and b["gain_db"] == 0.0)


def is_flat(cfg: dict | None) -> bool:
    """True when the EQ cannot change the signal: no config, bypassed, or no band that changes the sound
    (all disabled, or only gain-0 Peaking/Shelf bands) with input and output gain 0."""
    if cfg is None:
        return True
    g = cfg.get("global") or {}
    if g.get("bypass") is True:
        return True
    if float(g.get("input_gain", 0.0)) != 0.0 or float(g.get("output_gain", 0.0)) != 0.0:
        return False
    return not any(_band_changes_sound(b) for b in cfg.get("bands") or [])


def active_bands(cfg: dict | None) -> list[dict]:
    return [b for b in (cfg or {}).get("bands") or [] if _band_changes_sound(b)]


def resolve_path(voice: str = DEFAULT_VOICE, correction_eq_json: Path | str | None = None) -> Path:
    return Path(correction_eq_json) if correction_eq_json is not None else PROFILE_DIR / f"{voice}.json"


INDEX_FORMAT = "v8_correction_eq_index/1"
INDEX_PATH = PROFILE_DIR / "index.json"
SOURCES = ("explicit", "song_index", "voice", "auto", "none")


def song_index_path(song: str, index_path: Path | str = INDEX_PATH) -> Path | None:
    """Per-song correction EQ file from the index ({"format": INDEX_FORMAT, "songs": {<song folder name>:
    <file relative to the index>}}); None when there is no index or no entry. A listed file that does
    not exist raises."""
    ip = Path(index_path)
    if not ip.exists():
        return None
    try:
        d = json.loads(ip.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CorrectionEQError(f"{ip}: not valid JSON ({e})") from None
    if not isinstance(d, dict) or d.get("format") != INDEX_FORMAT or not isinstance(d.get("songs"), dict):
        raise CorrectionEQError(f"{ip}: format must be {INDEX_FORMAT!r} with a 'songs' object")
    rel = d["songs"].get(song)
    if rel is None:
        return None
    p = ip.parent / str(rel)
    if not p.exists():
        raise CorrectionEQError(f"{ip}: entry for {song!r} points to missing file {p}")
    return p


def resolve_source(song: str, voice: str = DEFAULT_VOICE, correction_eq_json: Path | str | None = None,
                   index_path: Path | str = INDEX_PATH) -> tuple[str, Path | None, dict | None]:
    """(source, path, cfg) in precedence order: explicit file > per-song index entry > <voice>.json >
    "auto" (path and cfg None; the engine computes it or falls back to flat). An explicit file that does
    not exist gives cfg None (flat), as before."""
    if correction_eq_json is not None:
        p = Path(correction_eq_json)
        return "explicit", p, load(p)
    p = song_index_path(song, index_path)
    if p is not None:
        return "song_index", p, load(p)
    p = PROFILE_DIR / f"{voice}.json"
    if p.exists():
        return "voice", p, load(p)
    return "auto", None, None


def auto_config(lead: Any, sr: int, voice: str | None = DEFAULT_VOICE, gap_db: float | None = None) -> dict:
    """Correction EQ derived from the lead itself (v8.auto_eq.auto_profile), validated. Imported lazily."""
    from mix_automation.v8.auto_eq import auto_profile

    return validate(auto_profile(lead, sr, voice=voice, gap_db=gap_db))


def auto_high_gap_db(raw_lead: Any, raw_inst_mono: Any) -> float:
    """v8.auto_eq.high_gap_db on the unprocessed lead and inst (input of the broad high band)."""
    from mix_automation.v8.auto_eq import high_gap_db

    return float(high_gap_db(raw_lead, raw_inst_mono))


def load(path: Path | str) -> dict | None:
    """Validated config, or None when the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CorrectionEQError(f"{p}: not valid JSON ({e})") from None
    return validate(data)


def save(cfg: dict, path: Path | str, overwrite: bool = False) -> Path:
    """Write a validated config. Refuses to replace an existing file unless overwrite=True."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} exists (pass overwrite=True to replace it)")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(validate(cfg), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_entry(path: Path, cfg: dict | None) -> dict:
    """What mix_report.json records about the correction EQ."""
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path) if path.exists() else None,
        "flat": is_flat(cfg),
        "global": (cfg or {}).get("global", {}),
        "bands": active_bands(cfg),
    }


# ------------------------------------------------------------------ plugin side


def write_to_plugin(plug: Any, cfg: dict) -> None:
    """Write global parameters and every band (bands not in cfg are disabled); read-back checked by
    set_vst3_param_strict. Expects a freshly loaded plugin (parameters not in cfg stay at load-time values)."""
    from mix_automation.dsp.vst3_adapter import set_vst3_param_strict

    set_vst3_param_strict(plug, "bypass", bool(cfg["global"].get("bypass", False)))
    for k, v in cfg["global"].items():
        if k != "bypass":
            set_vst3_param_strict(plug, k, v)
    by_n = {b["band"]: b for b in cfg["bands"]}
    for n in range(1, N_BANDS + 1):
        if f"band_{n}_enabled" not in plug.parameters:
            raise KeyError(f"Persistent Q has no band {n}")
        b = by_n.get(n)
        if b is None:
            set_vst3_param_strict(plug, f"band_{n}_enabled", False)
            continue
        set_vst3_param_strict(plug, f"band_{n}_type", b["type"])
        set_vst3_param_strict(plug, f"band_{n}_placement", b["placement"])
        set_vst3_param_strict(plug, f"band_{n}_frequency", b["frequency_hz"])
        set_vst3_param_strict(plug, f"band_{n}_gain", b["gain_db"])
        set_vst3_param_strict(plug, f"band_{n}_q", b["q"])
        if b["slope_db_oct"] is not None:
            set_vst3_param_strict(plug, f"band_{n}_slope", b["slope_db_oct"])
        for k, v in b["extra"].items():
            set_vst3_param_strict(plug, f"band_{n}_{k}", v)
        set_vst3_param_strict(plug, f"band_{n}_enabled", b["enabled"])


def _value(plug: Any, key: str) -> Any:
    """Plain Python value (pedalboard returns wrapper objects: WrappedBool is not a bool)."""
    v = getattr(plug, key)
    t = getattr(plug.parameters[key], "type", None)
    if t is bool:
        return bool(v)
    if t is str:
        return str(v)
    with contextlib.suppress(TypeError, ValueError):
        return round(float(v), 4)
    return str(v)


def fresh_reference() -> dict[str, Any]:
    """Parameter values of a freshly loaded Persistent Q (the "default" bands_from_plugin compares to)."""
    from pedalboard import load_plugin

    from mix_automation.config.plugin_registry import get_plugin_map

    ref = load_plugin(str(get_plugin_map()["persistent_q"].loadable_path))
    return {k: _value(ref, k) for k in ref.parameters}


def bands_from_plugin(plug: Any, reference: dict[str, Any] | None = None, voice: str | None = None,
                      source: dict | None = None) -> dict:
    """Correction EQ dict from a Persistent Q instance: global parameters and band extras that differ
    from `reference` (default: a freshly loaded Persistent Q), and every enabled band."""
    ref = reference if reference is not None else fresh_reference()
    g = {}
    for k in GLOBAL_PARAMS:
        if k in plug.parameters:
            v = _value(plug, k)
            if v != ref.get(k):
                g[k] = v
    bands = []
    for n in range(1, N_BANDS + 1):
        pre = f"band_{n}_"
        if pre + "enabled" not in plug.parameters or _value(plug, pre + "enabled") is not True:
            continue
        typ = _value(plug, pre + "type")
        slope = _value(plug, pre + "slope")
        extra = {}
        for s in EXTRA_SUFFIXES:
            if pre + s in plug.parameters:
                v = _value(plug, pre + s)
                if v != ref.get(pre + s):
                    extra[s] = v
        bands.append({
            "band": n, "enabled": True, "type": typ,
            "frequency_hz": float(_value(plug, pre + "frequency")),
            "gain_db": float(_value(plug, pre + "gain")),
            "q": float(_value(plug, pre + "q")),
            "slope_db_oct": float(slope) if (typ in PASS_TYPES or slope != ref.get(pre + "slope")) else None,
            "placement": _value(plug, pre + "placement"),
            "extra": extra,
        })
    return validate({"format": FORMAT, "voice": voice, "plugin": PLUGIN, "source": source or {},
                     "global": g, "bands": bands})


# ------------------------------------------------------------------ B4: averaging several settings (not used by the engine)

# 20 Hz - 20 kHz at 1/24 octave (R5 part B instruction): 241 points.
GRID_LO_HZ, GRID_HI_HZ, GRID_PER_OCT = 20.0, 20000.0, 24
RESPONSE_FS = 48000.0  # rate the biquads are evaluated at (bilinear warping near Nyquist depends on it)
AVERAGE_TYPES = ("Peaking", "LowShelf", "HighShelf")


def log_grid(lo: float = GRID_LO_HZ, hi: float = GRID_HI_HZ, per_oct: int = GRID_PER_OCT):
    import numpy as np

    n = math.ceil(np.log2(hi / lo) * per_oct) + 1  # both ends included; step just under 1/per_oct oct
    return np.geomspace(lo, hi, n)


def _biquad_db(typ: str, f0: float, gain_db: float, q: float, f, fs: float):
    """RBJ Audio EQ Cookbook peaking / shelf magnitude (dB) at f; shelves use the Q form of alpha."""
    import numpy as np

    a = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * min(f0, 0.49 * fs) / fs
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    if typ == "Peaking":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        den = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    else:
        sa = 2 * np.sqrt(a) * alpha
        if typ == "LowShelf":
            b = [a * ((a + 1) - (a - 1) * cw + sa), 2 * a * ((a - 1) - (a + 1) * cw), a * ((a + 1) - (a - 1) * cw - sa)]
            den = [(a + 1) + (a - 1) * cw + sa, -2 * ((a - 1) + (a + 1) * cw), (a + 1) + (a - 1) * cw - sa]
        else:  # HighShelf
            b = [a * ((a + 1) + (a - 1) * cw + sa), -2 * a * ((a - 1) + (a + 1) * cw), a * ((a + 1) + (a - 1) * cw - sa)]
            den = [(a + 1) - (a - 1) * cw + sa, 2 * ((a - 1) - (a + 1) * cw), (a + 1) - (a - 1) * cw - sa]
    z = np.exp(-1j * 2 * np.pi * np.asarray(f, dtype=float) / fs)
    h = (b[0] + b[1] * z + b[2] * z * z) / (den[0] + den[1] * z + den[2] * z * z)
    return 20 * np.log10(np.abs(h))


def _check_averageable(cfg: dict) -> None:
    g = cfg.get("global") or {}
    if float(g.get("input_gain", 0.0)) != 0.0:
        raise CorrectionEQError("average_profiles: input_gain is not supported")
    for b in active_bands(cfg):
        ex = b["extra"] or {}
        if b["type"] not in AVERAGE_TYPES or ex.get("dynamic") or ex.get("spectral"):
            raise CorrectionEQError(f"average_profiles: band {b['band']} ({b['type']}, extra {ex}) is not a "
                                    f"static {AVERAGE_TYPES} band")
        if b["placement"] != "Stereo":
            raise CorrectionEQError(f"average_profiles: band {b['band']} placement {b['placement']} is not Stereo")


def response_db(cfg: dict | None, f=None, fs: float = RESPONSE_FS):
    """Static magnitude response (dB) of a correction EQ: its active Peaking / shelf bands (RBJ) plus
    output gain. None / flat -> zeros."""
    import numpy as np

    f = log_grid() if f is None else np.asarray(f, dtype=float)
    out = np.zeros_like(f)
    if cfg is None or is_flat(cfg):
        return out
    _check_averageable(cfg)
    for b in active_bands(cfg):
        out += _biquad_db(b["type"], b["frequency_hz"], b["gain_db"], b["q"], f, fs)
    return out + float((cfg.get("global") or {}).get("output_gain", 0.0))


def _fit_bands(target, f, n_bands: int, fs: float, tol_db: float = 0.02) -> list[tuple[str, float, float, float]]:
    """Greedy (best of Peaking / LowShelf / HighShelf on the residual), each step refitting all bands."""
    import numpy as np
    from scipy.optimize import least_squares

    lf = np.log(f)
    lo = [np.log(FREQ_RANGE_HZ[0]), GAIN_RANGE_DB[0], np.log(Q_RANGE[0])]
    hi = [np.log(min(FREQ_RANGE_HZ[1], 0.45 * fs)), GAIN_RANGE_DB[1], np.log(10.0)]

    def model(types, p):
        y = np.zeros_like(f)
        for i, t in enumerate(types):
            y += _biquad_db(t, float(np.exp(p[3 * i])), float(p[3 * i + 1]), float(np.exp(p[3 * i + 2])), f, fs)
        return y

    def fit(types, p0):
        r = least_squares(lambda p: model(types, p) - target, p0, bounds=(lo * len(types), hi * len(types)))
        return r.x, float(np.sqrt(np.mean(r.fun ** 2)))

    types: list[str] = []
    p = np.zeros(0)
    for _ in range(n_bands):
        res = target - model(types, p)
        if np.sqrt(np.mean(res ** 2)) < tol_db:
            break
        k = int(np.argmax(np.abs(res)))
        cands = {"Peaking": [lf[k], res[k], 0.0], "LowShelf": [np.log(200.0), res[0], np.log(0.7)],
                 "HighShelf": [np.log(8000.0), res[-1], np.log(0.7)]}
        best = None
        for t, init in cands.items():
            x, err = fit([*types, t], np.concatenate([p, np.clip(init, lo, hi)]))
            if best is None or err < best[2]:
                best = (t, x, err)
        types.append(best[0])
        p = best[1]
    return [(t, float(np.exp(p[3 * i])), float(p[3 * i + 1]), float(np.exp(p[3 * i + 2]))) for i, t in enumerate(types)]


def average_profiles(profiles: list[dict | None], n_bands: int = 8, fs: float = RESPONSE_FS,
                     voice: str | None = None) -> dict:
    """Average several correction EQ settings: mean of their dB responses on the log grid (20 Hz-20 kHz,
    1/24 oct), refitted with up to `n_bands` Peaking / shelf bands. Static Stereo bands only (pass filters,
    dynamic and spectral bands raise). Output gain is folded into the bands. Not used by the engine (R5 B4)."""
    import numpy as np

    if not profiles:
        raise CorrectionEQError("average_profiles: no profiles")
    cfgs = [None if p is None else validate(p) for p in profiles]
    f = log_grid()
    target = np.mean([response_db(c, f, fs) for c in cfgs], axis=0)
    bands = []
    if np.max(np.abs(target)) >= 0.01:
        for i, (t, f0, g, q) in enumerate(_fit_bands(target, f, n_bands, fs)):
            if abs(g) >= 0.01:
                bands.append({"band": i + 1, "type": t, "frequency_hz": round(f0, 2), "gain_db": round(g, 2),
                              "q": round(q, 3)})
    out = validate({"format": FORMAT, "voice": voice, "plugin": PLUGIN, "bands": bands,
                    "source": {"averaged_from": len(cfgs), "method": "mean dB response, 1/24-oct grid, RBJ refit"}})
    out["source"]["fit_rms_db"] = round(float(np.sqrt(np.mean((response_db(out, f, fs) - target) ** 2))), 4)
    return out


__all__ = [
    "DEFAULT_VOICE",
    "FORMAT",
    "INDEX_FORMAT",
    "INDEX_PATH",
    "PROFILE_DIR",
    "SOURCES",
    "CorrectionEQError",
    "active_bands",
    "auto_config",
    "average_profiles",
    "bands_from_plugin",
    "file_sha256",
    "fresh_reference",
    "is_flat",
    "load",
    "log_grid",
    "report_entry",
    "resolve_path",
    "resolve_source",
    "response_db",
    "save",
    "song_index_path",
    "validate",
    "write_to_plugin",
]
