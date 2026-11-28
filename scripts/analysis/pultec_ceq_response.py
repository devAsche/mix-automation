"""Combined high-frequency response: R5 Pultec (real plugin, measured with noise) x correction EQ (RBJ).

    python scripts/analysis/pultec_ceq_response.py [--brightness 1.5] [--air 10000]

Pultec as in chain.lead_tone (100 CPS boost 2 / atten 3, hf boost = brightness at the air KCS, hf_q 4.5),
measured as H = Pxy / Pxx (Welch, white noise at -20 dBFS RMS), normalised to 0 dB at 1 kHz.
Correction EQ: static RBJ of each profile's unmuted bands, and of Ryo's Eva band 9 (8.3 kHz Q 0.6) alone.
Prints mean / min / max dB over 5-16 kHz and a 1/3-octave row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SR, print_table  # noqa: E402

THIRDS = [5000, 6300, 8000, 10000, 12500, 16000]


def pultec_db(f: np.ndarray, brightness: float, air_hz: float) -> np.ndarray:
    from scipy import signal

    from mix_automation.config.plugin_registry import get_plugin_map
    from mix_automation.dsp.vst3_adapter import configure_uad_pultec, set_vst3_param_strict
    from mix_automation.v8.chain import load_stage_plugin, run_plugin
    from mix_automation.v8.knobs import pultec_air_kcs

    hf_freq, _ = pultec_air_kcs(air_hz)
    plug, _ = load_stage_plugin(get_plugin_map()["uad_pultec_eqp1a"])
    configure_uad_pultec(plug, lf_boost=2.0, hf_boost=brightness, lf_freq="100 CPS", hf_freq=hf_freq, hf_q=4.5)
    set_vst3_param_strict(plug, "lf_atten", 3.0)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 20) * 0.1).astype(np.float32)
    y = np.asarray(run_plugin(plug, x, SR))
    y = y.mean(axis=0) if y.ndim == 2 else y
    fr, pxy = signal.csd(x[SR:], y[SR:], fs=SR, nperseg=16384)
    _, pxx = signal.welch(x[SR:], fs=SR, nperseg=16384)
    h = 20 * np.log10(np.abs(pxy / pxx) + 1e-12)
    h -= np.interp(1000.0, fr, h)
    return np.interp(f, fr, h)


def static_db(cfg: dict, f: np.ndarray) -> np.ndarray:
    """RBJ sum of the unmuted static bands (dynamic / spectral bands at their static gain)."""
    from mix_automation.v8.correction_eq import RESPONSE_FS, _biquad_db

    out = np.zeros_like(f)
    for b in cfg["bands"]:
        if (b["enabled"] and not (b.get("extra") or {}).get("mute") and b["gain_db"] != 0.0
                and b["type"] in ("Peaking", "LowShelf", "HighShelf")):
            out += _biquad_db(b["type"], b["frequency_hz"], b["gain_db"], b["q"], f, RESPONSE_FS)
    return out


def main() -> int:
    from mix_automation.v8 import correction_eq as ceq

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brightness", type=float, default=1.5)
    ap.add_argument("--air", type=float, default=10000.0)
    a = ap.parse_args()
    f = np.geomspace(5000, 16000, 200)
    pul = pultec_db(f, a.brightness, a.air)
    eva = ceq.load(ceq.PROFILE_DIR / "ryo_エバ_20261128.json")
    ld = ceq.load(ceq.PROFILE_DIR / "ryo_LD_20261128.json")
    b9 = {**eva, "bands": [b for b in eva["bands"] if abs(b["frequency_hz"] - 8322.74) < 1]}
    curves = {"Pultec": pul}
    for name, cfg in (("Eva 8.3k Q0.6 only", b9), ("Eva CEQ (all)", eva), ("L&D CEQ (all)", ld)):
        r = static_db(cfg, f)
        r1k = float(static_db(cfg, np.array([1000.0]))[0])
        curves[name] = r - r1k
        curves["Pultec + " + name] = pul + r - r1k
    rows = []
    for k, c in curves.items():
        rows.append([k, f"{np.mean(c):+.2f}", f"{np.min(c):+.2f}", f"{np.max(c):+.2f}",
                     *[f"{np.interp(t, f, c):+.1f}" for t in THIRDS]])
    print(f"dB re 1 kHz, 5-16 kHz (Pultec brightness {a.brightness} @ {a.air:.0f} Hz)")
    print_table(["curve", "mean", "min", "max", *[f"{t / 1000:g}k" for t in THIRDS]], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
