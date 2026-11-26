"""Phase 2: can the v8 metrics find the user's listening issues? (docs/evaluations/issues.yaml)

For each render (default: the four v7 renders in data/archive/gemini_v7_20261126) every metric is
computed (cached), then each issue is judged by the metric(s) of its category:

- Global issues: the song-level value against the pro distribution (5-95th percentile,
  docs/evaluations/reference_stats.json) or, for processing deltas, against the user's own mixes.
- Local issues: the window value against an absolute reference (pro phrase distribution / user's own
  mixes / psychoacoustic JND) and against the rest of the same song (10th/90th percentile).
  Window: +-1 s, or +-3 s when the time is approximate; "m:ss~ / から" extends the end by 3 s.
- "n_similar": how many places in the same song are at least as bad (reference only; unlisted
  places are not counted as false alarms because the user did not list everything).

Thresholds never come from issues.yaml. Song verdict: FAIL if any global rule fires.

    .venv\\Scripts\\python.exe scripts\\analysis\\evaluate_issues.py [render_dir ...] [--write-md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, fmt_time, print_table  # noqa: E402
from _v8eval import analyze_render  # noqa: E402
from build_reference_stats import V7_DIRS, scalars  # noqa: E402

from mix_automation.v8.metrics.steps import LEVEL_JND_DB  # noqa: E402
from mix_automation.v8.metrics.timing import DROPOUT_MIN_S  # noqa: E402

ARCHIVE = ROOT / "data" / "archive" / "gemini_v7_20261126"
ISSUES = ROOT / "docs" / "evaluations" / "issues.yaml"
REF = ROOT / "docs" / "evaluations" / "reference_stats.json"
MD = ROOT / "docs" / "evaluations" / "metric_validation.md"
EXTEND_FROM_S = 3.0

# Global rules: (rule id, category it speaks to, scalar key, side, label)
GLOBAL_RULES = [
    ("G_buried", "buried", "loss_median", "low", "部分ラウドネス損失（0.4 s ブロック中央値）"),
    ("G_sits_loss", "sits_on_top", "loss_median", "high", "同上が小さすぎる（マスクされなさすぎ）"),
    ("G_sits_narrow", "sits_on_top", "side_mid", "low", "ボーカル Side/Mid（狭さ）"),
    ("G_thin", "thin_light", "body_rel", "low", "(V-I)250-500 − (V-I)1-4k"),
    ("G_muddy", "muddy", "bright_rel", "low", "(V-I)4-16k − (V-I)250-500"),
    ("G_airy", "airy_fizzy", "air_rel", "high", "(V-I)8-16k − (V-I)1-4k"),
    ("G_airy_abs", "airy_fizzy", "hf_rel_abs", "high", "ボーカル自身 8-16k − 1-4k"),
    ("G_space_tail", "space_excess", "tail_ratio", "high", "フレーズ後 0.5 s の残り（dB）"),
    ("G_space_decay", "space_excess", "decay_t60", "high", "フレーズ終わりの減衰時間（T20→60 dB）"),
    ("G_space_vs_inst", "space_excess", "decay_over_inst_rt", "high", "減衰時間 ÷ オケのブラインド RT60"),
]
CATEGORY_RULES = {}
for r in GLOBAL_RULES:
    CATEGORY_RULES.setdefault(r[1], []).append(r)


def load_ref() -> dict:
    if not REF.exists():
        raise SystemExit(f"{REF.relative_to(ROOT)} missing: run scripts/analysis/build_reference_stats.py first")
    return json.loads(REF.read_text(encoding="utf-8"))


def pro_threshold(ref: dict, key: str, side: str, exclude: str | None = None) -> float | None:
    rows = ref["pro"]["rows"]
    vals = [r[key] for s, r in rows.items() if s != exclude and r.get(key) is not None]
    if len(vals) < 5:
        return None
    return float(np.percentile(vals, 5 if side == "low" else 95))


def fires(v: float | None, thr: float | None, side: str) -> bool | None:
    if v is None or thr is None:
        return None
    return v < thr if side == "low" else v > thr


def window_of(issue: dict) -> tuple[float, float, float]:
    t = float(issue.get("start", issue.get("time_sec")))
    tol = 3.0 if issue.get("approx") else 1.0
    end = float(issue["end"]) if "end" in issue else t
    ext = EXTEND_FROM_S if re.search(r"～|から", issue.get("quote", "").split("」")[0][:20]) else 0.0
    return t - tol, end + tol + ext, t


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def song_pct(values, q: float) -> float | None:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return float(np.percentile(v, q)) if v.size else None


def fmt(x, nd: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "Y" if x else "N"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def yn(x) -> str:
    return "—" if x is None else ("HIT" if x else "miss")


# --------------------------------------------------------------------------- local judges


def judge_buried(res, w0, w1, ref):
    bl = res["A"].get("pre_blocks") or res["A"]["blocks"]
    inwin = [b for b in bl if w0 <= b[0] < w1 and b[1] is not None]
    if not inwin:
        return None
    v = float(np.median([b[1] for b in inwin]))
    thr_abs = ref["pro"]["block_masking_loss"]["p5"]
    allv = [b[1] for b in bl if b[1] is not None]
    thr_rel = song_pct(allv, 10)
    sim = sum(1 for x in allv if x <= v)
    vi = float(np.median([b[2] for b in inwin]))
    return dict(metric="A 部分ラウドネス損失(phon, 窓内 0.4 s ブロック中央値)", value=v, abs_thr=thr_abs, abs_src="プロ全ブロック p5",
                rel_thr=thr_rel, hit_abs=v < thr_abs, hit_rel=v < thr_rel, n_similar=f"{sim}/{len(allv)} ブロック",
                extra=f"1-4k V/I {vi:.1f} dB")


def judge_colour_local(res, w0, w1, t_iss, ref, key="bright_rel", side="low"):
    tl = res["B"]["timeline"]
    t = np.array(tl["t"], dtype=float)
    vals = np.array([np.nan if x is None else x for x in tl[key]], dtype=float)
    sel = (t >= w0) & (t < w1) & np.isfinite(vals)
    if not np.any(sel):
        return None
    v = float(np.mean(vals[sel]))
    thr_rel = song_pct(vals, 10 if side == "low" else 90)
    after = vals[(t >= t_iss) & np.isfinite(vals)]
    before = vals[(t < t_iss) & np.isfinite(vals)]
    d = float(np.mean(after) - np.mean(before)) if after.size and before.size else None
    sim = int(np.sum(vals <= v)) if side == "low" else int(np.sum(vals >= v))
    hit_abs = (d is not None and d <= -LEVEL_JND_DB) if side == "low" else (d is not None and d >= LEVEL_JND_DB)
    return dict(metric=f"B {key} (6 s 窓)", value=v, abs_thr=-LEVEL_JND_DB, abs_src="以降−以前 の差 ≤ −1 dB(JND)",
                rel_thr=thr_rel, hit_abs=hit_abs, hit_rel=fires(v, thr_rel, side), n_similar=f"{sim}/{int(np.isfinite(vals).sum())} 窓",
                extra=f"以降−以前 {fmt(d)} dB")


def judge_consonant_raw(res, w0, w1, ref, kind, direction):
    """Same events, delta measured against the raw input lead (before v7 de-ess / RX)."""
    ev = [c for c in res.get("C_raw", []) if c[0] == kind]
    pv = {(c[1], c[2]): c for c in res.get("C_prep_vs_raw", []) if c[0] == kind}
    inwin = [c for c in ev if overlaps(c[1], c[2], w0, w1) and c[3] is not None]
    if not inwin:
        return dict(metric=f"C {kind} Δ子音/母音 raw基準", value=None, note="raw 入力なし / 窓内に検出なし")
    allv = [c[3] for c in ev if c[3] is not None]
    if direction == "crushed":
        c = min(inwin, key=lambda c: c[3])
        thr_rel = song_pct(allv, 10)
        hit_rel = fires(c[3], thr_rel, "low")
        sim = sum(1 for x in allv if x <= c[3])
    else:
        c = max(inwin, key=lambda c: c[3])
        thr_rel = song_pct(allv, 90)
        hit_rel = fires(c[3], thr_rel, "high")
        sim = sum(1 for x in allv if x >= c[3])
    pr = pv.get((c[1], c[2]))
    return dict(metric=f"C {kind} Δ子音/母音 raw基準(dB)", value=c[3], abs_thr=None, abs_src="（自MIX の raw 基準 Δ は未計算）",
                rel_thr=thr_rel, hit_abs=None, hit_rel=hit_rel, n_similar=f"{sim}/{len(allv)} 個",
                extra=f"@{fmt_time(c[1])} うち下処理(prepared−raw) {fmt(pr[3] if pr else None, 1)} dB")


def judge_consonant(res, w0, w1, ref, kind, direction):
    ev = [c for c in res.get("C", []) if c[0] == kind]
    inwin = [c for c in ev if overlaps(c[1], c[2], w0, w1) and c[3] is not None]
    if not inwin:
        return dict(metric=f"C {kind} Δ子音/母音", value=None, note="窓内に検出なし")
    sm = ref["self_mix"]["consonants"][kind]["delta_ratio"]
    allv = [c[3] for c in ev if c[3] is not None]
    if direction == "crushed":
        c = min(inwin, key=lambda c: c[3])
        v, thr_abs, thr_rel = c[3], sm["p5"] if sm else None, song_pct(allv, 10)
        hit_abs, hit_rel = fires(v, thr_abs, "low"), fires(v, thr_rel, "low")
        sim = sum(1 for x in allv if x <= v)
    else:
        c = max(inwin, key=lambda c: c[3])
        v, thr_abs, thr_rel = c[3], sm["p95"] if sm else None, song_pct(allv, 90)
        hit_abs, hit_rel = fires(v, thr_abs, "high"), fires(v, thr_rel, "high")
        sim = sum(1 for x in allv if x >= v)
    vi_ref = ref["pro"]["consonants"][kind]["vi"]
    return dict(metric=f"C {kind} Δ子音/母音(dB)", value=v, abs_thr=thr_abs, abs_src="自MIX Δ " + ("p5" if direction == "crushed" else "p95"),
                rel_thr=thr_rel, hit_abs=hit_abs, hit_rel=hit_rel, n_similar=f"{sim}/{len(allv)} 個",
                extra=f"@{fmt_time(c[1])} V/I {fmt(c[4], 1)} dB (プロ p5 {fmt(vi_ref['p5'] if vi_ref else None, 1)}), Δcrest {fmt(c[5], 1)}")


def judge_step(res, w0, w1, ref):
    d = res["D"]
    t = np.array(d["t"], dtype=float)
    far = np.array([np.nan if x is None else x for x in d["far"]], dtype=float)
    sel = (t >= w0) & (t < w1) & np.isfinite(far)
    if not np.any(sel):
        return None
    i = int(np.nanargmax(np.where(sel, far, -np.inf)))
    v = float(far[i])
    thr_rel = song_pct(far, 90)
    # distinct peaks (>= 3 s apart) at least as large
    order = np.argsort(-np.nan_to_num(far, nan=-np.inf))
    peaks = []
    for k in order:
        if not np.isfinite(far[k]) or far[k] < v:
            break
        if all(abs(t[k] - u) >= 3.0 for u in peaks):
            peaks.append(t[k])
    comp = f"Δlevel {fmt(d['d_level'][i], 1)} / Δbright {fmt(d['d_bright'][i], 1)} / Δwet {fmt(d['d_wet'][i], 1)} @{fmt_time(t[i])}"
    return dict(metric="D 段差 far_score(dB, 前後 3 s, リード推定)", value=v, abs_thr=LEVEL_JND_DB, abs_src="音量 JND 1 dB",
                rel_thr=thr_rel, hit_abs=v >= LEVEL_JND_DB, hit_rel=v >= thr_rel, n_similar=f"{len(peaks)} 箇所", extra=comp)


def judge_delay(res, w0, w1, ref):
    ds = [x for x in res["F"]["delays"] if overlaps(x["start"], x["end"] + 0.5, w0, w1)]
    if not ds:
        return dict(metric="F ディレイ対拍格子", value=None, note="窓内に飛ばしイベントなし")
    x = ds[0]
    return dict(metric="F ディレイ時間 − 最寄り音価(ms)", value=x["err"] * 1000, abs_thr=x["tol"] * 1000,
                abs_src="Friberg & Sundberg 1995: max(6 ms, 2.5% IOI)", rel_thr=None, hit_abs=x["off_grid"], hit_rel=None,
                n_similar=f"{sum(1 for y in res['F']['delays'] if y['off_grid'])}/{len(res['F']['delays'])} 飛ばし",
                extra=(f"delay {x['delay_nominal'] * 1000:.0f} ms ({x['note']}, 拍 {x['beat_ioi'] * 1000:.0f} ms); "
                       f"実測エコー周期 {fmt(x['echo_period'] * 1000 if x['echo_period'] else None, 0)} ms → 誤差 {fmt(x['echo_err'] * 1000 if x['echo_err'] is not None else None, 0)} ms"))


def judge_dropout(res, w0, w1, ref):
    g = [x for x in res["G"] if overlaps(x["start"], x["end"] + 0.5, w0, w1)]
    if not g:
        return dict(metric="G 演出抜け", value=None, note="窓内に演出イベントなし")
    x = max(g, key=lambda x: x["dropout"])
    return dict(metric="G 演出終了→音符終了(s)", value=x["dropout"], abs_thr=DROPOUT_MIN_S, abs_src="時間積分 ~100 ms (Zwicker & Fastl)",
                rel_thr=None, hit_abs=x["missing"], hit_rel=None, n_similar=f"{sum(1 for y in res['G'] if y['missing'])}/{len(res['G'])} 演出",
                extra="; ".join(f"{y['fx']} {fmt_time(y['start'])}–{fmt_time(y['end'])} → 音符終了 {fmt_time(y['note_end'])} ({y['dropout']:+.2f} s)" for y in g))


def judge_harmony(res, w0, w1, ref, field="prom"):
    """field 'prom': partial-loudness prominence (phon); 'rel_db': layer level minus lead level (dB)."""
    best = None
    for name, h in res.get("H", {}).items():
        if field not in h:
            continue
        t = np.array(h["t"], dtype=float)
        p = np.array([np.nan if x is None else x for x in h[field]], dtype=float)
        s = np.array([np.nan if x is None else x for x in h["semis"]], dtype=float)
        sel = (t >= w0) & (t < w1) & np.isfinite(p)
        if sel.sum() < 3:
            continue
        v = float(np.percentile(p[sel], 90))
        thr = song_pct(p, 90)
        semis = float(np.nanmedian(s[sel])) if np.isfinite(s[sel]).any() else None
        tt, pp = t[np.isfinite(p)], p[np.isfinite(p)]
        bins = np.floor(tt).astype(int)
        sim = sum(1 for b in np.unique(bins) if np.percentile(pp[bins == b], 90) >= v)
        label = "顕著度 p90 (phon" if field == "prom" else "対リード音量 p90 (dB"
        cand = dict(metric=f"H ハモリ{label}, {name})", value=v, abs_thr=None, abs_src="参照分布なし",
                    rel_thr=thr, hit_abs=None, hit_rel=fires(v, thr, "high"), n_similar=f"{sim} 秒（この層）",
                    extra=f"リードからの音程 {fmt(semis, 1)} 半音（pYIN、オクターブ誤りあり得る）")
        if best is None or v > best["value"]:
            best = cand
    return best or dict(metric="H ハモリ", value=None, note="窓内にハモリ層なし")


def judge_local(res, issue, ref):
    w0, w1, t_iss = window_of(issue)
    cat, d = issue["category"], issue["direction"]
    if cat == "buried":
        return [judge_buried(res, w0, w1, ref)]
    if cat == "muddy":
        return [judge_colour_local(res, w0, w1, t_iss, ref, "bright_rel", "low")]
    if cat == "sibilance":
        return [judge_consonant(res, w0, w1, ref, "sibilant", d), judge_consonant_raw(res, w0, w1, ref, "sibilant", d)]
    if cat == "consonant_attack":
        return [judge_consonant(res, w0, w1, ref, "attack", d), judge_consonant_raw(res, w0, w1, ref, "attack", d)]
    if cat == "step_change":
        return [judge_step(res, w0, w1, ref)]
    if cat == "delay_offbeat":
        return [judge_delay(res, w0, w1, ref)]
    if cat == "fx_dropout":
        return [judge_dropout(res, w0, w1, ref)]
    if cat == "harmony_prominent":
        return [judge_harmony(res, w0, w1, ref, "prom"), judge_harmony(res, w0, w1, ref, "rel_db")]
    return [dict(metric="(対応指標なし)", value=None, note="未対応カテゴリ")]


def judge_global(res, issue, ref, song_scalars):
    cat = issue["category"]
    out = []
    rules = CATEGORY_RULES.get(cat, [])
    if cat == "airy_fizzy" and issue["direction"] == "fizzy":
        fz = res["B"].get("fizz") or {}
        sm = [v["fizz"]["hf_rel_delta"] for v in ref["self_mix"]["fizz"].values() if v.get("fizz") and v["fizz"].get("hf_rel_delta") is not None]
        thr = max(sm) if sm else None
        out.append(dict(metric="B fizz: Δ(8-16k − 1-4k) 処理後−処理前 (dB)", value=fz.get("hf_rel_delta"), abs_thr=thr,
                        abs_src="自MIX の同じ差の最大", rel_thr=None, hit_abs=fires(fz.get("hf_rel_delta"), thr, "high"), hit_rel=None,
                        n_similar="—", extra=f"Δflatness {fmt(fz.get('flat_delta'), 3)}"))
        rules = [r for r in rules if r[0] == "G_airy_abs"]
    for rid, _, key, side, label in rules:
        v = song_scalars.get(key)
        thr = pro_threshold(ref, key, side)
        out.append(dict(metric=f"{rid}: {label}", value=v, abs_thr=thr, abs_src=f"プロ {'p5' if side == 'low' else 'p95'}",
                        rel_thr=None, hit_abs=fires(v, thr, side), hit_rel=None, n_similar="—"))
    if not out:
        out.append(dict(metric="(対応指標なし)", value=None, note="未対応カテゴリ"))
    return out


# --------------------------------------------------------------------------- main


def render_dirs(args) -> list[Path]:
    if args.renders:
        return [Path(p) for p in args.renders]
    return [ARCHIVE / d for d in V7_DIRS.values()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("renders", nargs="*")
    ap.add_argument("--write-md", action="store_true")
    ap.add_argument("--compute-only", action="store_true")
    args = ap.parse_args()

    results = {}
    for d in render_dirs(args):
        res = analyze_render(d)
        results[res["song"]] = res
        print(f"computed {res['song']}", flush=True)
    if args.compute_only:
        return
    ref = load_ref()
    issues = yaml.safe_load(ISSUES.read_text(encoding="utf-8"))["issues"]
    song_sc = {s: scalars(r) for s, r in results.items()}

    rows = []
    for iss in issues:
        res = results.get(iss["song"])
        if res is None:
            continue
        if iss.get("polarity") == "positive":
            continue
        js = judge_global(res, iss, ref, song_sc[iss["song"]]) if iss.get("scope") == "global" else judge_local(res, iss, ref)
        for j in js:
            if j is None:
                j = dict(metric="—", value=None, note="窓内にデータなし")
            hit = None
            if j.get("hit_abs") is not None or j.get("hit_rel") is not None:
                hit = bool(j.get("hit_abs")) or bool(j.get("hit_rel"))
            rows.append({"id": iss["id"], "cat": f"{iss['category']}/{iss['direction']}", "when": "全体" if iss.get("scope") == "global" else fmt_time(iss["time_sec"]), **j, "hit": hit})

    # Global rules on every song (verdict) + negatives (FiL distance OK, self mixes, pro leave-one-out).
    verdict_rows, fa_rows = [], []
    for song, sc in song_sc.items():
        fired = []
        for rid, cat, key, side, label in GLOBAL_RULES:
            f = fires(sc.get(key), pro_threshold(ref, key, side), side)
            if f:
                fired.append(f"{rid}({fmt(sc.get(key))})")
        verdict_rows.append([song, "FAIL" if fired else "PASS", ", ".join(fired) or "—"])
    for rid, cat, key, side, label in GLOBAL_RULES:
        pro_rows = ref["pro"]["rows"]
        loo = [fires(r.get(key), pro_threshold(ref, key, side, exclude=s), side) for s, r in pro_rows.items()]
        loo = [x for x in loo if x is not None]
        selfs = {s: fires(r.get(key), pro_threshold(ref, key, side), side) for s, r in ref["self_mix"]["rows"].items()}
        fil = fires(song_sc.get("Fall in Love", {}).get(key), pro_threshold(ref, key, side), side) if cat in ("buried", "sits_on_top", "space_excess") else None
        v7sep = {s: fires(r.get(key), pro_threshold(ref, key, side), side) for s, r in ref.get("v7_separated", {}).items()}
        fa_rows.append([rid, f"{sum(loo)}/{len(loo)}", " ".join(f"{s[:4]}:{'発火' if v else ('なし' if v is not None else '—')}" for s, v in selfs.items()),
                        "—" if fil is None else ("誤報" if fil else "OK"),
                        " ".join(f"{s[:3]}:{'発火' if v else 'なし'}" for s, v in v7sep.items() if v is not None)])

    # Per-metric hit rates.
    by_metric: dict[str, list[bool]] = {}
    for r in rows:
        key = r["metric"].split(":")[0].split(" (")[0].split("(")[0].strip()
        if r["hit"] is not None:
            by_metric.setdefault(key, []).append(r["hit"])

    hdr = ["id", "分類/向き", "時刻", "指標", "値", "絶対閾値", "根拠", "曲内閾値", "判定", "同程度", "補足"]
    table = [[r["id"], r["cat"], r["when"], r["metric"], fmt(r.get("value")), fmt(r.get("abs_thr")), r.get("abs_src", "—"),
              fmt(r.get("rel_thr")), yn(r["hit"]) + ("" if r["hit"] is None else f"({'abs' if r.get('hit_abs') else ''}{'+' if r.get('hit_abs') and r.get('hit_rel') else ''}{'rel' if r.get('hit_rel') else ''})" if r["hit"] else ""),
              r.get("n_similar", "—"), r.get("extra") or r.get("note", "")] for r in rows]
    print_table(hdr, table)
    print()
    print_table(["指標", "hit/件"], [[k, f"{sum(v)}/{len(v)}"] for k, v in by_metric.items()])
    print()
    print_table(["曲", "判定", "発火した全体ルール"], verdict_rows)
    print()
    print_table(["全体ルール", "プロ LOO 発火", "自MIX", "FiL(距離OK)", "v7 分離版"], fa_rows)

    # Colour ranking across songs (the user's statements: FiL brightest/airiest, シックス next,
    # ラブ&デストロイ most muffled). Order is only reported, not used for thresholds.
    rank_rows = []
    for key in ("air_rel", "hf_rel_abs", "bright_rel", "body_rel"):
        order = sorted(song_sc, key=lambda s: -(song_sc[s].get(key) or -99))
        rank_rows.append([key, " > ".join(f"{s[:4]}({fmt(song_sc[s].get(key), 1)})" for s in order)])
    print()
    print_table(["指標(大きい順)", "曲の並び"], rank_rows)

    # Beat This vs v7 tempo.
    tempo = []
    for s, r in results.items():
        f = r["F"]
        tempo.append([s, fmt(f.get("bpm_beatthis"), 1), fmt(f.get("bpm_v7"), 1), json.dumps(f.get("tempo_segments_v7")), fmt(r["duration"], 1)])
    print()
    print_table(["曲", "Beat This BPM", "v7 BPM", "v7 tempo_segments", "曲長(s)"], tempo)

    if args.write_md:
        write_md(table, hdr, by_metric, verdict_rows, fa_rows, tempo, results, ref, rank_rows)


def md_table(hdr, rows) -> str:
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def write_md(table, hdr, by_metric, verdict_rows, fa_rows, tempo, results, ref, rank_rows) -> None:
    parts = [
        "# 指標の検証（Phase 2）",
        "",
        "`scripts/analysis/evaluate_issues.py --write-md` の出力。指標の定義と出典は `src/mix_automation/v8/metrics/`、"
        "参照分布は `reference_stats.json`（`scripts/analysis/build_reference_stats.py`）。",
        f"参照: プロ {ref['pro']['n_songs']} 曲（BS-RoFormer 分離済みステム）、自MIX {len(ref['self_mix']['rows'])} 件。",
        "",
        "## 指摘 × 指標",
        "",
        "判定: HIT(abs) = 絶対基準（プロ分布・自MIX・JND）を超えた、HIT(rel) = 同じ曲の残りの時間の 10/90 パーセンタイルを超えた。"
        "時刻窓は ±1 s（approx は ±3 s）、「～／から」は終わりを +3 s。閾値は issues.yaml からは決めていない。"
        "「同程度」= 同じ曲で同じくらい悪い箇所の数（指摘されていない箇所は誤報に数えない）。",
        "",
        md_table(hdr, table),
        "",
        "## 指標ごとの hit 率",
        "",
        md_table(["指標", "hit/件"], [[k, f"{sum(v)}/{len(v)}"] for k, v in by_metric.items()]),
        "",
        "## 曲ごとの合否（全体ルールのどれかが発火したら FAIL）",
        "",
        md_table(["曲", "判定", "発火した全体ルール"], verdict_rows),
        "",
        "## 全体ルールの誤報（否定例）",
        "",
        "プロ LOO = その曲を除いた分布の p5/p95 で判定（構造上 約 10%）。自MIX・FiL（fil-g2: 距離感 OK）で発火したら誤報。"
        "v7 分離版 = v7 マスターを同じ BS-RoFormer で分離して計算（真のステムとの差 = 分離バイアス）。",
        "",
        md_table(["全体ルール", "プロ LOO 発火", "自MIX", "FiL(距離OK)", "v7 分離版"], fa_rows),
        "",
        "## 曲間の声色の並び（期待: Fall in Love が最も明るい／エアリー、シックスが次、ラブ&デストロイが最もこもる）",
        "",
        md_table(["指標(大きい順)", "曲の並び"], rank_rows),
        "",
        "## Beat This! と v7 のテンポ",
        "",
        md_table(["曲", "Beat This BPM", "v7 BPM", "v7 tempo_segments", "曲長(s)"], tempo),
        "",
    ]
    MD.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
