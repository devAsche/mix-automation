"""R4: search the v8 knobs per song against the pro references (two-stage CMA-ES).

    .venv\\Scripts\\python.exe scripts\\match\\run_match.py エバ --budget-min 120
    .venv\\Scripts\\python.exe scripts\\match\\run_match.py エバ --max-evals 3      # smoke test (no knobs JSON)
    .venv\\Scripts\\python.exe scripts\\match\\run_match.py エバ --resume           # continue an interrupted run
    .venv\\Scripts\\python.exe scripts\\match\\run_match.py エバ --finalize-only    # knobs JSON + LOO from the jsonl

Writes data/processed/v8r_match/<song>/{regions.json, scales.json, tempo.json, front_*.npz, evals*.jsonl,
result*.json} and data/outputs_v8r4/knobs/<song>.json (new files only). Uses the GPU (tempo, once) and the
real VST3 plugins: run one song at a time, never next to a render.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mix_automation.v8 import chain  # noqa: E402
from mix_automation.v8.knobs import V8Knobs, r3_knobs  # noqa: E402
from mix_automation.v8.match import (  # noqa: E402
    SONG_REFS,
    STAGE_A,
    STAGE_B,
    SongMatch,
    append_jsonl,
    best_record,
    from_unit,
    loo_select,
    read_jsonl,
    to_unit,
    write_knobs_json,
)

KNOBS_OUT = ROOT / "data" / "outputs_v8r4" / "knobs"
ALIASES = {"eva": "エバ", "love": "ラブ&デストロイ", "fall": "Fall in Love"}
CMA_SIGMA = 0.22  # same initial step as dsp/vst3_optimizer.py (CMA on [0,1]-normalised parameters)
SHARE_A = 0.8  # stage A is ~12x slower per evaluation (R4 smoke: 65 s vs 5.5 s); B still gets ~250 evaluations


def table(head: list[str], rows: list[list]) -> None:
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(head)] if rows else [len(h) for h in head]
    print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(head)))
    for r in rows:
        print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


class Runner:
    def __init__(self, sm: SongMatch, jsonl: Path, records: list[dict], max_evals: int | None) -> None:
        self.sm, self.jsonl, self.records = sm, jsonl, records
        self.max_evals = max_evals
        self.new = 0
        self.best_a_tone: tuple[float, np.ndarray] | None = None

    def left(self) -> int:
        return 10**9 if self.max_evals is None else self.max_evals - self.new

    def ev(self, knobs: V8Knobs, stage: str, label: str, tone: np.ndarray | None = None) -> dict:
        res, tone_out = self.sm.evaluate(knobs, tone)
        rec = {"i": len(self.records), "stage": stage, "label": label, "knobs": knobs.to_dict(), **res,
               "t": time.strftime("%Y-%m-%d %H:%M:%S")}
        append_jsonl(self.jsonl, rec)
        self.records.append(rec)
        self.new += 1
        if stage in ("A", "default") and label != "R3_default" and np.isfinite(res["dist"]) and \
                (self.best_a_tone is None or res["dist"] < self.best_a_tone[0]):
            self.best_a_tone = (res["dist"], tone_out)
        pc = " ".join(f"{c[:8]}={v:.2f}" for c, v in res["per_cover"].items())
        print(f"[{stage}:{label}] dist={res['dist']:.3f} ({pc}) {res['sec']:.0f}s", flush=True)
        return rec

    def cma(self, stage: str, names: tuple[str, ...], base: V8Knobs, n: int, seed: int,
            tone: np.ndarray | None) -> None:
        from cmaes import CMA

        if n <= 0:
            return
        opt = CMA(mean=to_unit(base, names), sigma=CMA_SIGMA, bounds=np.array([[0.0, 1.0]] * len(names)), seed=seed)
        done = 0
        while done < n and self.left() > 0:
            sols = []
            for _ in range(opt.population_size):
                if done >= n or self.left() <= 0:
                    break
                u = opt.ask()
                rec = self.ev(from_unit(u, names, base), stage, f"g{opt.generation}", tone)
                d = rec["dist"] if np.isfinite(rec["dist"]) else 1e6
                sols.append((u, d))
                done += 1
            if len(sols) == opt.population_size:
                opt.tell(sols)


def knobs_of(rec: dict) -> V8Knobs:
    return replace(V8Knobs(), **{k: float(v) for k, v in rec["knobs"].items()})


def summarize(sm: SongMatch, records: list[dict], jsonl: Path, write: bool) -> dict:
    final = best_record(records, ("A", "B", "delay")) or best_record([r for r in records if r["label"] == "R4_default"])
    pool = [r for r in records if r["stage"] != "default"]
    r3 = next((r for r in records if r["label"] == "R3_default"), None)
    r4 = next((r for r in records if r["label"] == "R4_default"), None)
    out: dict = {"song": sm.song, "covers": sm.covers, "jsonl": str(jsonl), "n_evals": len(records),
                 "scale_source": sm.scale_source,
                 "final": {"index": final["i"], "dist": final["dist"], "per_cover": final["per_cover"],
                           "knobs": final["knobs"]} if final else None}
    rows = []
    for lab, r in (("R3_default", r3), ("R4_default", r4), ("R4_match", final)):
        if r:
            rows.append([lab, round(r["dist"], 3)] + [round(r["per_cover"].get(c, float("nan")), 3) for c in sm.covers])
    table(["setting", "median"] + sm.covers, rows)
    if len(sm.covers) >= 2 and pool:
        loo = loo_select(pool, sm.covers)
        out["loo"] = {"note": "candidates = evaluated records (no extra render); see docs/v8r_match/README.md",
                      "folds": loo,
                      "R3_default": {c: r3["per_cover"].get(c) for c in sm.covers} if r3 else None,
                      "R4_default": {c: r4["per_cover"].get(c) for c in sm.covers} if r4 else None}
        table(["held_out", "picked_i", "train_med", "held_out_dist", "R3_def", "R4_def"],
              [[c, f["index"], f["train_median"], f["held_out"],
                round(r3["per_cover"][c], 3) if r3 else "", round(r4["per_cover"][c], 3) if r4 else ""]
               for c, f in loo.items()])
    if write and final:
        p = write_knobs_json(knobs_of(final), KNOBS_OUT, sm.song)
        out["knobs_json"] = str(p)
        print(f"knobs -> {p}")
    rp = sm.dir / f"result_{time.strftime('%Y%m%d-%H%M%S')}.json"
    rp.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"result -> {rp}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("song", help="エバ / ラブ&デストロイ / Fall in Love (or eva / love / fall)")
    ap.add_argument("--budget-min", type=float, default=120.0, help="wall-clock budget for this run (minutes)")
    ap.add_argument("--max-evals", type=int, default=None, help="cap on evaluations in this run (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="append to evals.jsonl and start from its best")
    ap.add_argument("--finalize-only", action="store_true", help="no evaluation: knobs JSON + LOO from the jsonl")
    args = ap.parse_args()
    song = ALIASES.get(args.song.lower(), args.song)
    if song not in SONG_REFS:
        raise SystemExit(f"unknown song {args.song!r}; R4 songs: {list(SONG_REFS)}")

    t_start = time.time()
    budget_s = args.budget_min * 60.0
    sm = SongMatch(song)
    sm.prepare()
    info = json.loads((sm.dir / "regions.json").read_text(encoding="utf-8"))
    table(["#", "t0", "t1", "len_s", "lead_share", "rank"],
          [[i, r["t0"], r["t1"], round(r["t1"] - r["t0"], 2), r["lead_share"], r["rank"]]
           for i, r in enumerate(info["regions"])])
    print(f"regions={len(info['regions'])} eval_s={info['eval_s']} excerpt_s={info['excerpt_s']} "
          f"candidates={info['n_candidates']} tempo={sm.tempo_bpm:.2f} scales={sm.scale_source} "
          f"prep={time.time() - t_start:.0f}s")

    jsonl = sm.dir / "evals.jsonl"
    if jsonl.exists() and not (args.resume or args.finalize_only):
        jsonl = sm.dir / f"evals_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"  # never append to an old run by accident
    records = read_jsonl(jsonl)
    if args.finalize_only:
        summarize(sm, records, jsonl, write=True)
        return

    run = Runner(sm, jsonl, records, args.max_evals)
    r4 = V8Knobs()
    with chain.plugin_cache():
        have = {r["label"] for r in records}
        for lab, kn in (("R3_default", r3_knobs()), ("R4_default", r4)):
            if lab not in have and run.left() > 0:
                run.ev(kn, "default", lab)
        t_a = float(np.mean([r["sec"] for r in records if r["stage"] in ("default", "A")] or [0.0]))
        # Stage-B probe on the R4-default lead: measures the cheap evaluation and is a valid B candidate.
        if run.left() > 0:
            if run.best_a_tone is None:
                run.best_a_tone = (float("inf"), sm.tone(r4))
            run.ev(r4, "B", "probe", run.best_a_tone[1])
        t_b = float(np.mean([r["sec"] for r in records if r["stage"] in ("B", "delay")] or [t_a]))
        remaining = budget_s - (time.time() - t_start) - 2 * t_b  # two delay types are evaluated at the end
        n_a = max(0, int(SHARE_A * remaining / max(t_a, 1e-6)))
        n_b = max(0, int((1 - SHARE_A) * remaining / max(t_b, 1e-6)))
        print(f"eval_s A={t_a:.1f} B={t_b:.1f} -> planned A={n_a} B={n_b} (budget {args.budget_min:.0f} min)")

        start_a = best_record(records, ("A",))
        base_a = knobs_of(start_a) if start_a else r4
        run.cma("A", STAGE_A, base_a, n_a, args.seed, None)
        best_a = best_record(records, ("A",))
        r4_rec = next((r for r in records if r["label"] == "R4_default"), None)
        if best_a is None or (r4_rec and r4_rec["dist"] <= best_a["dist"]):
            best_a = r4_rec
        if best_a is None or run.left() <= 0:
            print(f"stopped after {run.new} evaluations ({time.time() - t_start:.0f}s); no knobs JSON written")
            summarize(sm, records, jsonl, write=False)
            return
        base_b = knobs_of(best_a)
        if run.best_a_tone is None or not np.isclose(run.best_a_tone[0], best_a["dist"]):
            run.best_a_tone = (best_a["dist"], sm.tone(base_b))
        tone = run.best_a_tone[1]
        run.cma("B", STAGE_B, base_b, n_b, args.seed + 1, tone)
        best_b = best_record([r for r in records if r["stage"] in ("B",) and r["label"] != "probe"]) or best_a
        base_d = knobs_of(best_b)
        for dt in (0, 1, 2):
            if dt != int(base_d.delay_type) and run.left() > 0:
                run.ev(replace(base_d, delay_type=float(dt)), "delay", f"delay_type={dt}", tone)
    print(f"done: {run.new} evaluations in {time.time() - t_start:.0f}s")
    summarize(sm, records, jsonl, write=n_a > 0)


if __name__ == "__main__":
    main()
