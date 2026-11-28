"""Softube VCA Compressor: attack / release parameter (0.25-4.0, no unit shown) -> measured time constants.

    python scripts/analysis/vca_time_constants.py [--values 0.25 0.4 1 2 4]

A 1 kHz tone steps from -30 to -6 dBFS (held 1 s) and back; ratio 4, threshold -20 dBFS. The gain
(output / input envelope, 2 ms RMS) is fitted per step: attack = time to 63 % of the final gain
reduction after the up-step, release = time to 63 % recovery after the down-step (ms).
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


def _env_db(x: np.ndarray, win: int) -> np.ndarray:
    k = np.ones(win) / win
    return 10 * np.log10(np.convolve(x.astype(np.float64) ** 2, k, mode="same") + 1e-20)


def _t63(g: np.ndarray, i0: int, i1: int) -> float:
    seg = g[i0:i1]
    start, end = seg[0], np.median(seg[-int(0.1 * SR):])
    tgt = start + 0.63 * (end - start)
    idx = np.argmax((seg - tgt) * np.sign(end - start) >= 0)
    return idx / SR * 1000.0


def main() -> int:
    from mix_automation.config.plugin_registry import get_plugin_map
    from mix_automation.dsp.vst3_adapter import configure_softube_vca
    from mix_automation.v8.chain import load_stage_plugin, run_plugin

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--values", nargs="*", type=float, default=[0.25, 0.4, 1.0, 2.0, 4.0])
    a = ap.parse_args()
    t = np.arange(int(3.0 * SR)) / SR
    amp = np.where((t >= 1.0) & (t < 2.0), 10 ** (-6 / 20), 10 ** (-30 / 20))
    x = (np.sin(2 * np.pi * 1000 * t) * amp * np.sqrt(2)).astype(np.float32)
    win = int(0.002 * SR)
    plug, _ = load_stage_plugin(get_plugin_map()["vca_compressor"])
    rows = []
    for v in a.values:
        configure_softube_vca(plug, threshold_db=-20.0, ratio=4.0, attack_norm=v, release_norm=0.85,
                              sc_hpf_hz=120.0, output_db=0.0, dry_wet=100.0)
        ya = np.asarray(run_plugin(plug, x, SR))
        configure_softube_vca(plug, threshold_db=-20.0, ratio=4.0, attack_norm=0.4, release_norm=v,
                              sc_hpf_hz=120.0, output_db=0.0, dry_wet=100.0)
        yr = np.asarray(run_plugin(plug, x, SR))
        ga = _env_db(ya.mean(axis=0) if ya.ndim == 2 else ya, win) - _env_db(x, win)
        gr = _env_db(yr.mean(axis=0) if yr.ndim == 2 else yr, win) - _env_db(x, win)
        rows.append([f"{v:g}", f"{_t63(ga, SR, 2 * SR):.1f}", f"{_t63(gr, 2 * SR, 3 * SR - win):.1f}",
                     f"{np.median(ga[int(1.8 * SR):2 * SR]):+.1f}"])
    print_table(["value", "attack ms (63%)", "release ms (63%)", "GR at -6 dBFS (dB)"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
