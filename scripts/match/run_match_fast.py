"""R4b: the R4 two-stage CMA-ES search, evaluated in parallel processes with early stopping.

    .venv\\Scripts\\python.exe scripts\\match\\run_match_fast.py エバ --budget-min 15
    .venv\\Scripts\\python.exe scripts\\match\\run_match_fast.py ラブ&デストロイ --init-from エバ
    .venv\\Scripts\\python.exe scripts\\match\\run_match_fast.py エバ ラブ&デストロイ --shared-a
    .venv\\Scripts\\python.exe scripts\\match\\run_match_fast.py エバ --max-gens-a 1 --max-gens-b 1   # smoke

Flow as R4 (scripts/match/run_match.py, kept unchanged): R4_default and R4_match (data/outputs_v8r4/knobs)
are evaluated first, then stage A (STAGE_A), stage B (STAGE_B on the best stage-A lead) and the two other
delay types. One CMA generation is asked at once and evaluated on the pool. Distance = R4b definition
(tail cuts + AFx-Rep, SongMatch(r4b=True)); the R4 definition is logged as dist_r4.

Writes data/processed/v8r_match/<song>/{regions_r4b.json, scales_r4b.json, front_*.npz,
evals_r4b_<time>.jsonl, result_r4b_<time>.json} and data/outputs_v8r4b/knobs/<song>[ _sharedA].json
(new files only). Real VST3 plugins and the GPU: never next to a render or another matching run.

--r5 (補正 EQ ループ一周目): tone knobs fixed at knobs.R5_TONE, stage A = STAGE_A_R5 (dynamics only),
distance = SongMatch(r5=True) (WEIGHTS_R5; dist_r4b logged). Starting points: R4_default and R4_match
as they are (comparison only) and with R5_TONE ("+tone"); only R5_TONE candidates can be the final.
Writes data/processed/v8r_s1loop/<song>/{regions_r5.json, scales_r5.json, front_*.npz (when not
already in v8r_match), evals_r5_<time>.jsonl, result_r5_<time>.json} and data/outputs_v8r5/knobs/<song>.json.

    .venv\\Scripts\\python.exe scripts\\match\\run_match_fast.py エバ --r5 --budget-min 15
    .venv\\Scripts\\python.exe scripts\\match\\run_match_fast.py エバ --r5 --max-gens-a 1 --max-gens-b 1 --no-knobs  # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mix_automation.v8.knobs import R5_TONE, V8Knobs, has_r5_tone, load_knobs, with_r5_tone  # noqa: E402
from mix_automation.v8.match import (  # noqa: E402
    SONG_REFS,
    STAGE_A,
    STAGE_A_R5,
    STAGE_B,
    SongMatch,
    append_jsonl,
    from_unit,
    loo_select,
    tail_spot_terms,
    to_unit,
    write_knobs_json,
)
from mix_automation.v8.match_pool import DEFAULT_WORKERS, EvalPool, finite_or, should_stop  # noqa: E402

KNOBS_OUT = ROOT / "data" / "outputs_v8r4b" / "knobs"
KNOBS_R4 = ROOT / "data" / "outputs_v8r4" / "knobs"
KNOBS_OUT_R5 = ROOT / "data" / "outputs_v8r5" / "knobs"
ALIASES = {"eva": "エバ", "love": "ラブ&デストロイ", "fall": "Fall in Love"}
CMA_SIGMA = 0.22  # as R4 (dsp/vst3_optimizer.py)
# Time share of stage A in what is left after the starting evaluations: R4 Eva levelled off after
# ~20 A and ~60 B evaluations at 65 s / 5.5 s each (README 4 章, 6 章) -> 1300 s : 330 s = 0.8.
SHARE_A = 0.8


def table(head: list[str], rows: list[list]) -> None:
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(head)] if rows else [len(h) for h in head]
    print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(head)))
    for r in rows:
        print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def r(v: float, n: int = 3) -> float | str:
    return round(float(v), n) if v is not None and np.isfinite(v) else "nan"


@dataclass
class SongRun:
    sm: SongMatch
    jsonl: Path
    records: list[dict] = field(default_factory=list)
    tones: dict[int, np.ndarray] = field(default_factory=dict)  # record index -> tone (kept only when useful)

    def add(self, knobs: V8Knobs, stage: str, label: str, res: dict) -> dict:
        tone = res.pop("tone", None)
        rec = {"i": len(self.records), "stage": stage, "label": label, "knobs": knobs.to_dict(), **res,
               "t": time.strftime("%Y-%m-%d %H:%M:%S")}
        append_jsonl(self.jsonl, rec)
        self.records.append(rec)
        if tone is not None:
            self.tones[rec["i"]] = tone
        return rec


@dataclass
class Problem:
    """One CMA run: own on one song, or shared over several songs (objective = mean distance)."""
    stage: str
    names: tuple[str, ...]
    base: V8Knobs
    songs: list[str]
    seed: int
    start_obj: float
    tones: dict[str, np.ndarray | None]
    want_tone: bool
    best_obj: float = math.inf
    best_recs: dict[str, dict] = field(default_factory=dict)
    hist: list[float] = field(default_factory=list)
    gens: int = 0
    stopped: str = ""
    opt: object = None


class Search:
    def __init__(self, pool: EvalPool, runs: dict[str, SongRun], popsize: int) -> None:
        self.pool, self.runs, self.popsize = pool, runs, popsize
        self.stage_stats: dict[str, dict] = {}

    def _stat(self, stage: str, n: int, sec: float, gens: int = 0) -> None:
        s = self.stage_stats.setdefault(stage, {"n_evals": 0, "wall_s": 0.0, "gens": 0})
        s["n_evals"] += n
        s["wall_s"] = round(s["wall_s"] + sec, 1)
        s["gens"] += gens

    def evaluate(self, items: list[tuple[str, V8Knobs, str, str, np.ndarray | None, bool]]) -> list[dict]:
        """items = (song, knobs, stage, label, tone, want_tone) -> records (same order)."""
        t0 = time.time()
        outs = self.pool.map([(s, k, tone, wt) for s, k, _, _, tone, wt in items])
        recs = [self.runs[s].add(k, st, lab, res) for (s, k, st, lab, _, _), res in zip(items, outs, strict=True)]
        stages = {it[2] for it in items}
        for st in stages:
            n = sum(1 for it in items if it[2] == st)
            self._stat(st, n, (time.time() - t0) * n / len(items))
        for rec in recs:
            if "error" in rec:
                print(f"  ! {rec['stage']}:{rec['label']} {rec['error']}", flush=True)
        return recs

    def cma(self, probs: list[Problem], deadline: float, max_gens: int) -> None:
        from cmaes import CMA

        for p in probs:
            p.opt = CMA(mean=to_unit(p.base, p.names), sigma=CMA_SIGMA, bounds=np.array([[0.0, 1.0]] * len(p.names)),
                        seed=p.seed, population_size=self.popsize)
            p.best_obj = p.start_obj
            p.hist = [p.start_obj]
        last_gen_s = 0.0
        while True:
            active = [p for p in probs if not p.stopped]
            for p in active:
                if p.gens >= max_gens:
                    p.stopped = "max_gens"
                elif time.time() + last_gen_s > deadline:
                    p.stopped = "budget"
            active = [p for p in probs if not p.stopped]
            if not active:
                return
            t0 = time.time()
            items, index = [], []
            for p in active:
                us = [p.opt.ask() for _ in range(p.opt.population_size)]
                for j, u in enumerate(us):
                    kn = from_unit(u, p.names, p.base)
                    for s in p.songs:
                        items.append((s, kn, p.stage, f"g{p.gens}", p.tones.get(s), p.want_tone))
                        index.append((p, j, u, s))
            recs = self.evaluate(items)
            per: dict[int, dict] = {}
            for (p, j, u, s), rec in zip(index, recs, strict=True):
                per.setdefault(id(p), {}).setdefault(j, (u, {}))[1][s] = rec
            for p in active:
                sols = []
                for _j, (u, by_song) in sorted(per[id(p)].items()):
                    obj = float(np.mean([finite_or(by_song[s]["dist"]) for s in p.songs]))
                    sols.append((u, obj))
                    if obj < p.best_obj:
                        p.best_obj, p.best_recs = obj, dict(by_song)
                p.opt.tell(sols)
                p.gens += 1
                p.hist.append(p.best_obj)
                # free tones that are no longer the best of this problem
                keep = {rec["i"] for rec in p.best_recs.values()}
                for s in p.songs:
                    run = self.runs[s]
                    for i in [i for i in run.tones if i not in keep and run.records[i]["stage"] == p.stage]:
                        del run.tones[i]
                tag = "+".join(p.songs) if len(p.songs) > 1 else p.songs[0]
                print(f"[{p.stage} {tag}] gen {p.gens}: best {p.best_obj:.3f} (gen min "
                      f"{min(o for _, o in sols):.3f}) {time.time() - t0:.0f}s", flush=True)
                if should_stop(p.hist):
                    p.stopped = "converged"
            last_gen_s = time.time() - t0
            self._stat(probs[0].stage, 0, 0.0, gens=1)


def knobs_of(rec: dict) -> V8Knobs:
    return replace(V8Knobs(), **{k: float(v) for k, v in rec["knobs"].items()})


def resolve_init(arg: str | None, dirs: tuple[Path, ...] = (KNOBS_OUT, KNOBS_R4)) -> V8Knobs | None:
    if not arg:
        return None
    p = Path(arg)
    if p.suffix == ".json" and p.exists():
        return load_knobs(p)
    song = ALIASES.get(arg.lower(), arg)
    for d in dirs:
        if (d / f"{song}.json").exists():
            print(f"init-from: {d / f'{song}.json'}")
            return load_knobs(d / f"{song}.json")
    raise SystemExit(f"--init-from: no knobs for {arg!r} in {' or '.join(str(d) for d in dirs)}")


def tail_summary(run: SongRun, rec: dict | None) -> dict:
    """Per spot: v8 tail, and per cover the pro tail, pro decay (tail - late) and two/one-sided; plus medians."""
    if rec is None:
        return {}
    v8 = {"tail_by_spot": rec["v8"].get("tail_by_spot"), "late_by_spot": rec["v8"].get("late_by_spot")}
    spots, med = [], {}
    for c in run.sm.covers:
        rows = tail_spot_terms(v8, run.sm.pro[c]) if v8["tail_by_spot"] is not None else []
        for i, at, bt, dec, kind in rows:
            spots.append({"spot": i, "cover": c, "v8_tail_db": round(at, 2), "pro_tail_db": round(bt, 2),
                          "pro_decay_db": round(dec, 2), "kind": kind})
        if rows:
            med[c] = {"v8_tail_db": round(float(np.median([x[1] for x in rows])), 2),
                      "pro_tail_db": round(float(np.median([x[2] for x in rows])), 2),
                      "pro_decay_db": round(float(np.median([x[3] for x in rows])), 2),
                      "n_two": sum(x[4] == "two" for x in rows), "n_one": sum(x[4] == "one" for x in rows)}
    return {"v8_tail_by_spot": v8["tail_by_spot"], "v8_late_by_spot": v8["late_by_spot"],
            "spots": spots, "median_by_cover": med}


def print_tail_table(run: SongRun, rec: dict, label: str) -> None:
    t = tail_summary(run, rec)
    by = {(x["spot"], x["cover"]): x for x in t.get("spots", [])}
    rows = []
    for i, v in enumerate(t.get("v8_tail_by_spot") or []):
        row = [i, r(v, 1)]
        for c in run.sm.covers:
            x = by.get((i, c))
            row.append(f"{x['pro_tail_db']:.1f}/{x['pro_decay_db']:.1f}/{x['kind']}" if x else "-")
        rows.append(row)
    print(f"tail per spot ({label}; pro = tail/decay/kind)")
    table(["spot", f"v8_{label}"] + run.sm.covers, rows)


def calib_means(recs: list[dict]) -> dict[str, dict[str, float]]:
    """Mean calibration renders per chain stage: starting evaluations (the first task of each worker, no
    warm start yet when they fit in one wave) vs stage A (warm-started)."""
    out: dict[str, dict[str, float]] = {}
    for grp, stages in (("start", ("start",)), ("A", ("A",))):
        vals: dict[str, list[int]] = {}
        for x in recs:
            if x["stage"] in stages:
                for st, n in (x.get("calib_evals") or {}).items():
                    vals.setdefault(st, []).append(n)
        for st, v in vals.items():
            out.setdefault(st, {})[grp] = round(float(np.mean(v)), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("songs", nargs="+", help="エバ / ラブ&デストロイ / Fall in Love (or eva / love / fall)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--budget-min", type=float, default=15.0, help="wall-clock budget (minutes, whole run)")
    ap.add_argument("--max-gens-a", type=int, default=30)
    ap.add_argument("--max-gens-b", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=None,
                    help="CMA population (default: cmaes' 4+3ln(10)=10 rounded up to a multiple of --workers)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None, help="song name or knobs.json used as the stage A/B start")
    ap.add_argument("--shared-a", action="store_true", help="one stage A for all songs (mean distance)")
    ap.add_argument("--r5", action="store_true",
                    help="R5: tone fixed at knobs.R5_TONE, stage A = dynamics only, WEIGHTS_R5 (dist_r4b logged)")
    ap.add_argument("--no-knobs", action="store_true", help="do not write the knobs JSON (smoke tests)")
    args = ap.parse_args()
    songs = [ALIASES.get(s.lower(), s) for s in args.songs]
    for s in songs:
        if s not in SONG_REFS:
            raise SystemExit(f"unknown song {s!r}; songs: {list(SONG_REFS)}")
    if args.shared_a and len(songs) < 2:
        raise SystemExit("--shared-a needs at least 2 songs")
    stage_a = STAGE_A_R5 if args.r5 else STAGE_A
    knobs_out = KNOBS_OUT_R5 if args.r5 else KNOBS_OUT
    pop = args.popsize or args.workers * math.ceil((4 + math.floor(3 * math.log(len(stage_a)))) / args.workers)

    t_start = time.time()
    deadline = t_start + args.budget_min * 60.0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs: dict[str, SongRun] = {}
    for s in songs:
        sm = SongMatch(s, r4b=True, r5=args.r5)
        sm.prepare()
        info = json.loads((sm.dir / f"regions{sm.sfx}.json").read_text(encoding="utf-8"))
        n_valid = {c: int(np.sum(np.isfinite(np.asarray(sm.pro[c].get("tail_by_spot") or [], float))))
                   for c in sm.covers}
        print(f"{s}: regions={len(info['regions'])} eval_s={info['eval_s']} tail_spots={len(info['tail_spots'])} "
              f"pro tail spots found={n_valid} scales={sm.scale_source}")
        runs[s] = SongRun(sm, sm.dir / f"evals{sm.sfx}_{stamp}.jsonl")
    t_prep = time.time() - t_start
    print(f"prepare {t_prep:.0f}s; workers={args.workers} popsize={pop}", flush=True)

    init = resolve_init(args.init_from, (KNOBS_OUT_R5, KNOBS_OUT, KNOBS_R4) if args.r5 else (KNOBS_OUT, KNOBS_R4))
    if args.r5 and init is not None:
        init = with_r5_tone(init)
    init_label = "init+tone" if args.r5 else "init"
    r4 = V8Knobs()
    with EvalPool({s: runs[s].sm for s in songs}, args.workers) as pool:
        t_pool = time.time()
        search = Search(pool, runs, pop)
        items = []
        for s in songs:
            items.append((s, r4, "start", "R4_default", None, True))
            if args.r5:
                items.append((s, with_r5_tone(r4), "start", "R4_default+tone", None, True))
            if (KNOBS_R4 / f"{s}.json").exists():
                r4m = load_knobs(KNOBS_R4 / f"{s}.json")
                items.append((s, r4m, "start", "R4_match", None, True))
                if args.r5:
                    items.append((s, with_r5_tone(r4m), "start", "R4_match+tone", None, True))
            if init is not None:
                items.append((s, init, "start", init_label, None, True))
        recs = search.evaluate(items)
        for it, rec in zip(items, recs, strict=True):
            extra = f"dist_r4b={r(rec.get('dist_r4b'))}" if args.r5 else f"dist_r4={r(rec.get('dist_r4'))}"
            print(f"  [{rec['label']} {it[0]}] dist={r(rec['dist'])} {extra} "
                  f"{rec['sec']:.0f}s calib={rec.get('calib_evals')}", flush=True)
        # stage A start: --init-from, else the better of R4_default and R4_match (mean over songs);
        # R5: only the R5_TONE versions are eligible
        if init is not None:
            start_label = init_label
        else:
            starts = ("R4_default+tone", "R4_match+tone") if args.r5 else ("R4_default", "R4_match")
            labs = [lab for lab in starts
                    if all(any(x["label"] == lab for x in runs[s].records) for s in songs)]
            start_label = min(labs, key=lambda lab: np.mean([finite_or(next(
                x for x in runs[s].records if x["label"] == lab)["dist"]) for s in songs]))
        start_rec = {s: next(x for x in runs[s].records if x["label"] == start_label) for s in songs}
        base_a = init if init is not None else knobs_of(start_rec[songs[0]])
        if args.r5 and not has_r5_tone(base_a):
            raise SystemExit("R5: stage A must start from R5_TONE")
        print(f"stage A starts from {start_label}", flush=True)
        a_deadline = time.time() + SHARE_A * (deadline - time.time())

        # ---- stage A
        if args.shared_a:
            probs_a = [Problem("A", stage_a, base_a, list(songs), args.seed,
                               float(np.mean([finite_or(start_rec[s]["dist"]) for s in songs])),
                               dict.fromkeys(songs), True, best_recs=dict(start_rec))]
        else:
            probs_a = [Problem("A", stage_a, base_a, [s], args.seed, finite_or(start_rec[s]["dist"]),
                               {s: None}, True, best_recs={s: start_rec[s]}) for s in songs]
        search.cma(probs_a, a_deadline, args.max_gens_a)
        best_a = {s: p.best_recs[s] for p in probs_a for s in p.songs}

        # ---- stage B (own per song, on the best stage-A lead)
        probs_b = []
        for s in songs:
            ba = best_a[s]
            tone = runs[s].tones.get(ba["i"])
            if tone is None:
                tone = runs[s].sm.tone(knobs_of(ba))  # not expected: every A/start evaluation returns its tone
            probs_b.append(Problem("B", STAGE_B, knobs_of(ba), [s], args.seed + 1, finite_or(ba["dist"]),
                                   {s: tone}, False, best_recs={s: ba}))
        search.cma(probs_b, deadline, args.max_gens_b)

        # ---- delay types
        items = []
        for p in probs_b:
            s = p.songs[0]
            bd = knobs_of(p.best_recs[s])
            for dt in (0, 1, 2):
                if dt != int(bd.delay_type):
                    items.append((s, replace(bd, delay_type=float(dt)), "delay", f"delay_type={dt}", p.tones[s], False))
        search.evaluate(items)
        t_search = time.time() - t_pool

    total = time.time() - t_start
    print(f"done in {total:.0f}s (prepare {t_prep:.0f}s, search {t_search:.0f}s)")
    for s in songs:
        run = runs[s]
        recs = run.records
        searched = [x for x in recs if x["stage"] in ("A", "B", "delay") and np.isfinite(x["dist"])]
        cand = [x for x in recs if x["stage"] in ("B", "delay") and np.isfinite(x["dist"])] + [best_a[s]]
        r4m_rec = next((x for x in recs if x["label"] == "R4_match"), None)
        if args.r5:
            # R5: R4_match+tone stays eligible; R4_match itself (other tone) is for comparison only
            r4mt = next((x for x in recs if x["label"] == "R4_match+tone"), None)
            if r4mt is not None and np.isfinite(r4mt["dist"]):
                cand.append(r4mt)
            cand = [x for x in cand if has_r5_tone(x["knobs"])]
        elif r4m_rec is not None and np.isfinite(r4m_rec["dist"]):
            cand.append(r4m_rec)  # the R4 result stays eligible as the final choice
        final = min(cand, key=lambda x: x["dist"])
        search_best = min(searched, key=lambda x: x["dist"]) if searched else None
        rows = []
        cmp_key = "dist_r4b" if args.r5 else "dist_r4"
        start_labs = (("R4_default", "R4_default+tone", "R4_match", "R4_match+tone", init_label) if args.r5
                      else ("R4_default", "R4_match", "init"))
        for lab, rec in ([(lab, next((x for x in recs if x["label"] == lab), None)) for lab in start_labs]
                         + [("best_A", best_a[s]), ("R5_final" if args.r5 else "R4b_final", final)]):
            if rec:
                rows.append([lab, rec["i"], r(rec["dist"]), r(rec.get(cmp_key)), r(rec["v8"].get("tail"), 2)]
                            + [r(rec["per_cover"].get(c, float("nan"))) for c in run.sm.covers])
        print(f"--- {s}")
        table(["setting", "i", "dist", cmp_key, "v8_tail"] + run.sm.covers, rows)
        table(["stage", "n_evals", "wall_s", "gens"],
              [[st, v["n_evals"], v["wall_s"], v["gens"]] for st, v in search.stage_stats.items()])
        r4m = next((x for x in recs if x["label"] == "R4_match"), None)
        for lab, rec in (("R4_match", r4m), ("final", final)):
            if rec:
                print_tail_table(run, rec, lab)
        out = {
            "song": s, "mode": "sharedA" if args.shared_a else "own", "covers": run.sm.covers,
            "jsonl": str(run.jsonl), "n_evals": len(recs), "workers": args.workers, "popsize": pop,
            "budget_min": args.budget_min, "max_gens": {"A": args.max_gens_a, "B": args.max_gens_b},
            "init_from": args.init_from, "scale_source": run.sm.scale_source,
            "definition": "r5" if args.r5 else "r4b", "stage_a_knobs": list(stage_a),
            "tone_fixed": R5_TONE if args.r5 else None,
            "stages": search.stage_stats, "stop": {"A": [p.stopped for p in probs_a if s in p.songs],
                                                   "B": [p.stopped for p in probs_b if s in p.songs]},
            "prepare_s": round(t_prep, 1), "search_s": round(t_search, 1), "total_s": round(total, 1),
            "baselines": {lab: {"i": rec["i"], "dist": rec["dist"], "dist_r4": rec.get("dist_r4"),
                                "dist_r4b": rec.get("dist_r4b"),
                                "per_cover": rec["per_cover"], "per_cover_r4": rec.get("per_cover_r4")}
                          for lab in start_labs
                          for rec in [next((x for x in recs if x["label"] == lab), None)] if rec},
            "best_A": {"i": best_a[s]["i"], "dist": best_a[s]["dist"]},
            "final": {"i": final["i"], "dist": final["dist"], "dist_r4": final.get("dist_r4"),
                      "dist_r4b": final.get("dist_r4b"), "label": final["label"],
                      "per_cover": final["per_cover"], "knobs": final["knobs"]},
            "tail": {"final": tail_summary(run, final), "R4_match": tail_summary(run, r4m)},
            "start_A": start_label,
            "search_best": {"i": search_best["i"], "dist": search_best["dist"]} if search_best else None,
            "search_beat_R4_match": (bool(search_best["dist"] < r4m_rec["dist"])
                                     if search_best and r4m_rec and np.isfinite(r4m_rec["dist"]) else None),
            "calib_evals_mean": calib_means(recs),
        }
        pool_recs = [x for x in recs if x["stage"] != "start" and np.isfinite(x["dist"])]
        if len(run.sm.covers) >= 2 and pool_recs:
            out["loo"] = loo_select(pool_recs, run.sm.covers)
        if any(x["stage"] in ("A", "B") for x in recs) and not args.no_knobs:
            name = f"{s}_sharedA" if args.shared_a else s
            kp = write_knobs_json(knobs_of(final), knobs_out, name)
            out["knobs_json"] = str(kp)
            print(f"knobs -> {kp}")
        rp = run.sm.dir / f"result{run.sm.sfx}_{stamp}.json"
        rp.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"result -> {rp}")


if __name__ == "__main__":
    main()
