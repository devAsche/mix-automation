"""R3: v7 "main vocal only" vs v8 on the same song (short table + level-matched listening files).

    python scripts/analysis/v8r3_compare.py --song エバ --v7 data/outputs_phase2/エバ_auto \
        --v8 data/outputs_v8r3/エバ_v8 --out data/outputs_v8r3

a) v7 main-only: debug_stems lead_dry + wet (vocal) and inst_unmasked (backing) through v7's own
   apply_2mix_bus_glue_and_safe_master(target -9 LUFS) -> <out>/v7_mainonly/<song>/<song>_v7main_{Master,VocalBus}.wav
b) voc_minus_inst_lu / vocal_side_mid_db / master_lufs for both, with v8.dsp.mix_metrics (same as mix_report).
   Vocal and backing are taken at pre-bus level on both sides (v8: stems/lead_dry + wet_*, inst_ducked).
c) both masters gain-matched to -14 LUFS -> <out>/listen/<song>_v7main.wav, <song>_v8.wav (refuses to clip).

Existing output files are never overwritten (the script stops instead).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SR, load_ch_first, print_table  # noqa: E402

LISTEN_LUFS = -14.0  # streaming reference level for level-matched listening


def _stereo(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a, a]) if a.shape[0] == 1 else a[:2]


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    return a[:, :n] if a.shape[1] >= n else np.pad(a, ((0, 0), (0, n - a.shape[1])))


def _write_new(path: Path, x: np.ndarray, subtype: str) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32).T, SR, subtype=subtype)


def v7_mainonly(v7_dir: Path, song: str, out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from mix_automation.dsp.vst3_optimizer import apply_2mix_bus_glue_and_safe_master
    from mix_automation.v8.dsp import integrated_lufs

    dbg = v7_dir / "debug_stems"
    inst = _stereo(load_ch_first(dbg / "inst_unmasked.wav"))
    n = inst.shape[1]
    vocal = _fit(_stereo(load_ch_first(dbg / "lead_dry.wav")), n) + _fit(_stereo(load_ch_first(dbg / "wet.wav")), n)
    master = apply_2mix_bus_glue_and_safe_master(inst, vocal, SR, target_lufs=-9.0)
    gain_db = integrated_lufs(master, SR) - integrated_lufs(inst + vocal, SR)  # same uniform-gain rule as v8 VocalBus
    d = out / "v7_mainonly" / song
    _write_new(d / f"{song}_v7main_Master.wav", master, "PCM_24")
    _write_new(d / f"{song}_v7main_VocalBus.wav", vocal * 10 ** (gain_db / 20), "FLOAT")
    return vocal, inst, master


def v8_parts(v8_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stems = v8_dir / "stems"
    inst = _stereo(load_ch_first(stems / "inst_ducked.wav"))
    n = inst.shape[1]
    vocal = _fit(_stereo(load_ch_first(stems / "lead_dry.wav")), n)
    for w in sorted(stems.glob("wet_*.wav")):
        vocal = vocal + _fit(_stereo(load_ch_first(w)), n)
    masters = sorted(v8_dir.glob("*_v8_Master.wav"))
    if not masters:
        raise SystemExit(f"no *_v8_Master.wav in {v8_dir}")
    return vocal, inst, _stereo(load_ch_first(masters[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--song", required=True)
    ap.add_argument("--v7", required=True, type=Path, help="v7 render folder with debug_stems/")
    ap.add_argument("--v8", required=True, type=Path, help="v8 render folder (<song>_v8)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from mix_automation.v8.dsp import integrated_lufs, mix_metrics

    rows = []
    listen = {}
    for label, (voc, inst, master) in (
        ("v7main", v7_mainonly(args.v7, args.song, args.out)),
        ("v8", v8_parts(args.v8)),
    ):
        m = mix_metrics(voc, inst, master, SR)
        rows.append([label, m["voc_minus_inst_lu"], m["vocal_side_mid_db"], m["master_lufs"]])
        g = LISTEN_LUFS - integrated_lufs(master, SR)
        lv = master * 10 ** (g / 20)
        peak = float(np.max(np.abs(lv)))
        if peak >= 1.0:
            raise SystemExit(f"{label}: listening copy would clip (peak {peak:.3f}); not written")
        listen[label] = (lv, round(g, 2), round(20 * np.log10(peak), 2))
    for label, (lv, _, _) in listen.items():
        _write_new(args.out / "listen" / f"{args.song}_{label}.wav", lv, "PCM_24")

    print_table(["render", "voc_minus_inst_lu", "vocal_side_mid_db", "master_lufs", "listen_gain_db", "listen_peak_dbfs"],
                [r + [listen[r[0]][1], listen[r[0]][2]] for r in rows])


if __name__ == "__main__":
    main()
