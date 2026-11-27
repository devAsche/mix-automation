"""R4b step 0: is it safe to run the real VST3 chain in several processes at once?

    .venv\\Scripts\\python.exe scripts\\match\\parallel_safety.py --levels 1 1 2 4 6

Each worker loads every plugin of the matching chain ONCE (chain.plugin_cache, PhoenixVerb excepted as in
R4), waits on a barrier so all workers load/process at the same moment, then renders the front end, the
tone chain (VCA / LA-2A / Pultec / RX / Saturation / kHs) and the sends (doubler / PhoenixVerb / PLATE-140)
on the same 15 s of the Eva excerpt: base knobs, other knobs, base knobs again (reuse + reset check).
A thread lists the visible windows owned by the worker's own process (iLok / licence dialogs).
Outputs are compared with the first serial run: null = 20 log10(rms(diff) / rms(ref)).
Writes only data/processed/v8r_match/parallel_safety/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SR = 44100
SEG_S = 15.0
OUT = ROOT / "data" / "processed" / "v8r_match" / "parallel_safety"


def _own_windows(pid: int) -> set[str]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    titles: set[str] = set()
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lp):
        p = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid and user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            titles.add(buf.value or "<untitled>")
        return True

    user32.EnumWindows(proto(cb), 0)
    return titles


def worker(args: tuple) -> dict:
    wid, barrier, front_path = args
    pid = os.getpid()
    seen: set[str] = set()
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            try:
                seen.update(_own_windows(pid))
            except Exception:  # noqa: BLE001 - the watcher must never kill the worker
                pass
            time.sleep(0.3)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    out: dict = {"wid": wid, "pid": pid, "error": None}
    t0 = time.time()
    try:
        from mix_automation.v8 import chain
        from mix_automation.v8.knobs import V8Knobs

        z = np.load(front_path)
        n = int(SEG_S * SR)
        x = z["x"][:n].astype(np.float32)
        base = V8Knobs()
        other = replace(base, comp_total_gr_db=9.0, brightness=4.5, deess_db=5.0, saturation=2.0, plate_decay_s=1.8)
        res: dict[str, np.ndarray] = {}
        if barrier is not None:
            barrier.wait(timeout=600)
        with chain.plugin_cache():
            fx, mask = chain.lead_front(x, SR, chain.StageLog())
            res["front"] = fx
            for tag, kn in (("base1", base), ("other", other), ("base2", base)):
                tone = chain.lead_tone(fx, mask, SR, kn, chain.StageLog())
                res[f"tone_{tag}"] = tone
                sends = chain.build_sends(tone, SR, kn, 120.0, chain.StageLog())
                for k, v in sends.items():
                    res[f"{k}_{tag}"] = v
        path = OUT / f"_w{wid}_{pid}.npz"
        np.savez(path, **res)
        out["npz"] = str(path)
    except Exception as e:  # noqa: BLE001 - report every failure
        out["error"] = f"{type(e).__name__}: {e}"
        out["trace"] = traceback.format_exc()[-1500:]
    out["sec"] = round(time.time() - t0, 1)
    time.sleep(1.0)
    stop.set()
    th.join(timeout=2)
    out["windows"] = sorted(seen)
    return out


def null_db(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n].astype(np.float64), b[..., :n].astype(np.float64)
    ref = np.sqrt(np.mean(b ** 2)) + 1e-20
    d = np.sqrt(np.mean((a - b) ** 2))
    return float(20 * np.log10(d / ref + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 1, 2, 4, 6])
    ap.add_argument("--song", default="エバ")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    front = sorted((ROOT / "data" / "processed" / "v8r_match" / args.song).glob("front_*.npz"))[0]
    ctx = mp.get_context("spawn")
    ref: dict[str, np.ndarray] | None = None
    report: list[dict] = []
    for run_i, n in enumerate(args.levels):
        mgr = ctx.Manager()
        barrier = mgr.Barrier(n) if n > 1 else None
        t0 = time.time()
        with ctx.Pool(n) as pool:
            outs = pool.map(worker, [(w, barrier, str(front)) for w in range(n)])
        wall = time.time() - t0
        mgr.shutdown()
        for o in outs:
            row = {"run": run_i, "procs": n, "wid": o["wid"], "sec": o["sec"], "error": o["error"],
                   "windows": o["windows"]}
            if o.get("npz"):
                z = dict(np.load(o["npz"]))
                os.remove(o["npz"])
                if ref is None:
                    ref = z
                row["null_vs_ref_db"] = {k: round(null_db(z[k], ref[k]), 1) for k in ref if k in z}
                row["reuse_base2_vs_base1_db"] = {k.replace("_base2", ""): round(null_db(z[k], z[k.replace("base2", "base1")]), 1)
                                                  for k in z if k.endswith("_base2")}
            else:
                row["trace"] = o.get("trace")
            report.append(row)
            worst = max(row.get("null_vs_ref_db", {"-": float("nan")}).values())
            worst_r = max(row.get("reuse_base2_vs_base1_db", {"-": float("nan")}).values())
            print(f"run{run_i} procs={n} w{o['wid']} sec={o['sec']} wall={wall:.0f} err={o['error']} "
                  f"windows={o['windows']} worst_null_vs_ref={worst} worst_reuse={worst_r}", flush=True)
    p = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"report -> {p}")


if __name__ == "__main__":
    main()
