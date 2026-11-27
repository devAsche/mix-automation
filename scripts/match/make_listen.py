"""R4: full-song v8 renders (R4 match vs R3 defaults) and level-matched listening files.

    .venv\\Scripts\\python.exe scripts\\match\\make_listen.py --render     # renders (one at a time), then files
    .venv\\Scripts\\python.exe scripts\\match\\make_listen.py              # files only, from existing renders

Renders: mix-auto mix data/raw/<song> --engine v8 --v8-knobs <json> -o data/outputs_v8r4/renders/<R4|R3default>
(R4 json = data/outputs_v8r4/knobs/<song>.json from run_match.py; R3 json = R3_DEFAULTS written next to it).
Listening files -> data/outputs_v8r4/listen/, gain only to -14 LUFS; when that gain would push the sample
peak over 0 dBFS the gain is lowered to a -1 dBFS sample peak instead (no limiter) and the table says so.
  <song>_a_pro_<cover>_ハモリ込み.wav  pro cover (with its harmonies), aligned to the off-vocal timeline (R2 drift)
  <song>_b_v8_R4.wav / <song>_c_v8_R3default.wav
  <song>_d_v7main.wav  v7 main-vocal-only master (scripts/analysis/v8r3_compare.py procedure)
Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from mix_automation.v8.engine import song_slug  # noqa: E402
from mix_automation.v8.knobs import r3_knobs  # noqa: E402
from mix_automation.v8.match import load_drift_curve, orig_to_cover, write_knobs_json  # noqa: E402

OUT = ROOT / "data" / "outputs_v8r4"
SR = 44100
LISTEN_LUFS = -14.0  # streaming reference level, as v8r3_compare.py
PEAK_CEIL_DBFS = -1.0  # fallback sample-peak ceiling when -14 LUFS would clip (R4 指示)
SONGS = {
    # pro cover for the listening file: lowest R2 subtraction residual of the song (README 2 章)
    "エバ": {"pair": "Ryo_Eva__cover_Ryushen", "v7": ROOT / "data/outputs_phase2/エバ_auto",
            "v7main_existing": ROOT / "data/outputs_v8r3/listen/エバ_v7main.wav"},
    "ラブ&デストロイ": {"pair": "Ryo_LoveAndDestroy__cover_WolpisCarter_gVT-",
                  "v7": ROOT / "data/outputs_phase2/ラブandデストロイ_auto", "v7main_existing": None},
}


def render(song: str, tag: str, knobs_json: Path) -> None:
    exe = ROOT / ".venv" / "Scripts" / "mix-auto.exe"
    cmd = [str(exe), "mix", str(ROOT / "data" / "raw" / song), "--engine", "v8", "--v8-knobs", str(knobs_json),
           "-o", str(OUT / "renders" / tag)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)  # sequential on purpose: GPU + real VST3


def newest_master(song: str, tag: str) -> Path:
    d = OUT / "renders" / tag
    cands = sorted(d.glob(f"{song_slug(song)}_v8*/*_v8_Master.wav"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no v8 render for {song} in {d}")
    return cands[-1]


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path}: sr {sr} != {SR}")
    x = x.T
    return np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x[:2]


def level(x: np.ndarray) -> tuple[np.ndarray, dict]:
    from mix_automation.v8.dsp import integrated_lufs

    li = integrated_lufs(x, SR)
    g = LISTEN_LUFS - li
    peak = 20 * np.log10(np.max(np.abs(x)) + 1e-12)
    note = ""
    if peak + g > 0.0:
        g = PEAK_CEIL_DBFS - peak
        note = f"peak-limited gain (sample peak {PEAK_CEIL_DBFS} dBFS)"
    y = x * 10 ** (g / 20)
    return y, {"gain_db": round(g, 2), "lufs": round(integrated_lufs(y, SR), 2),
               "peak_dbfs": round(20 * np.log10(np.max(np.abs(y)) + 1e-12), 2), "note": note}


def write_new(path: Path, x: np.ndarray) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.T.astype(np.float32), SR, subtype="PCM_24")


def pro_aligned(pair: str, n: int) -> np.ndarray:
    key = pair.removeprefix("Ryo_")
    cov = load(ROOT / "data" / "processed" / "v8_cover_refs" / "wav" / f"{key}.wav")
    s0 = round(float(orig_to_cover(0.0, load_drift_curve(pair))))
    out = np.zeros((2, n), np.float32)
    lo, hi = max(0, s0), min(cov.shape[1], s0 + n)
    out[:, lo - s0:hi - s0] = cov[:, lo:hi]
    return out


def v7main(song: str, cfg: dict) -> np.ndarray:
    if cfg["v7main_existing"] is not None:
        return load(cfg["v7main_existing"])
    from v8r3_compare import v7_mainonly

    d = OUT / "v7_mainonly" / song / f"{song}_v7main_Master.wav"
    if d.exists():
        return load(d)
    _, _, master = v7_mainonly(cfg["v7"], song, OUT)
    return np.asarray(master, np.float32)


def main() -> None:
    from _common import print_table

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render", action="store_true", help="run the two v8 renders per song first (sequential)")
    ap.add_argument("--songs", default=",".join(SONGS))
    args = ap.parse_args()
    songs = [s for s in args.songs.split(",") if s]
    if args.render:
        for song in songs:
            r4 = OUT / "knobs" / f"{song}.json"
            if not r4.exists():
                raise SystemExit(f"missing {r4}: run scripts/match/run_match.py {song} first")
            r3 = OUT / "knobs" / f"{song}_R3default.json"
            if not r3.exists():
                r3 = write_knobs_json(r3_knobs(), OUT / "knobs", f"{song}_R3default")
            render(song, "R4", r4)
            render(song, "R3default", r3)
    rows = []
    for song in songs:
        cfg = SONGS[song]
        b = load(newest_master(song, "R4"))
        c = load(newest_master(song, "R3default"))
        n = b.shape[1]
        files = {
            f"{song}_a_pro_{cfg['pair'].split('__cover_')[-1]}_ハモリ込み.wav": pro_aligned(cfg["pair"], n),
            f"{song}_b_v8_R4.wav": b,
            f"{song}_c_v8_R3default.wav": c,
            f"{song}_d_v7main.wav": v7main(song, cfg),
        }
        for name, x in files.items():
            y, st = level(x)
            write_new(OUT / "listen" / name, y)
            rows.append([name, st["gain_db"], st["lufs"], st["peak_dbfs"], st["note"]])
    print_table(["file", "gain_db", "lufs", "sample_peak_dbfs", "note"], rows)


if __name__ == "__main__":
    main()
