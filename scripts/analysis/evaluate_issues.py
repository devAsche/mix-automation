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
from _common import (  # noqa: E402
    ROOT,
    fmt_time,
    load_report,  # noqa: E402
    print_table,
)
from _v8eval import analyze_debug_render, analyze_render, asr_cer, true_pre_vocal  # noqa: E402
from build_reference_stats import V7_DIRS, scalars  # noqa: E402

from mix_automation.v8.metrics.steps import LEVEL_JND_DB  # noqa: E402
from mix_automation.v8.metrics.timing import DROPOUT_MIN_S  # noqa: E402

ARCHIVE = ROOT / "data" / "archive" / "gemini_v7_20261126"
# Phase 2b re-renders with --export-debug-stems (null to the archive: master -65 dB).
PHASE2_OUT = ROOT / "data" / "outputs_phase2"
PROC = ROOT / "data" / "processed"
ISSUES = ROOT / "docs" / "evaluations" / "issues.yaml"
REF = ROOT / "docs" / "evaluations" / "reference_stats.json"
MD = ROOT / "docs" / "evaluations" / "metric_validation.md"
EXTEND_FROM_S = 3.0

# Global rules (Phase 2 review): judged as the difference to the ORIGINAL (本家) of the same song,
# because the right value differs per song and that spread is what makes a pro-wide distribution
# useless. (rule id, category, scalar key, side, label, threshold, kind) - kind "db": a-b in dB,
# "ratio": a/b - 1.
LEVEL_JND = 1.0  # intensity / loudness-level JND ~0.5-1 dB (Zwicker & Fastl, Psychoacoustics, sec. 7.1)
RT_JND = 0.05  # reverberation (EDT) JND 5 % (ISO 3382-1:2009, Annex A, Table A.1)
GLOBAL_RULES = [
    ("R_buried", "buried", "loss_median", "low", "部分ラウドネス損失 v7−本家 (phon)", LEVEL_JND, "db"),
    ("R_sits_loss", "sits_on_top", "loss_median", "high", "部分ラウドネス損失 v7−本家（マスクされなさすぎ）", LEVEL_JND, "db"),
    ("R_sits_narrow", "sits_on_top", "side_mid", "low", "Side/Mid v7−本家 (dB)", LEVEL_JND, "db"),
    ("R_sits_dry", "sits_on_top", "tail_ratio", "low", "フレーズ後の残り v7−本家 (dB)", LEVEL_JND, "db"),
    ("R_thin", "thin_light", "body_rel", "low", "body_rel v7−本家 (dB)", LEVEL_JND, "db"),
    ("R_muddy", "muddy", "bright_rel", "low", "bright_rel v7−本家 (dB)", LEVEL_JND, "db"),
    ("R_airy", "airy_fizzy", "air_rel", "high", "air_rel v7−本家 (dB)", LEVEL_JND, "db"),
    ("R_airy_abs", "airy_fizzy", "hf_rel_abs", "high", "ボーカル自身 8-16k−1-4k v7−本家 (dB)", LEVEL_JND, "db"),
    ("R_space_tail", "space_excess", "tail_ratio", "high", "フレーズ後の残り v7−本家 (dB)", LEVEL_JND, "db"),
    ("R_space_decay", "space_excess", "decay_t60", "high", "減衰時間 v7/本家−1", RT_JND, "ratio"),
]
CATEGORY_RULES: dict[str, list] = {}
for r in GLOBAL_RULES:
    CATEGORY_RULES.setdefault(r[1], []).append(r)


def rel_delta(a: dict | None, b: dict | None, key: str, kind: str) -> float | None:
    if not a or not b or a.get(key) is None or b.get(key) is None:
        return None
    if kind == "ratio":
        return a[key] / b[key] - 1.0 if b[key] else None
    return a[key] - b[key]


def fires_rel(d: float | None, thr: float, side: str) -> bool | None:
    if d is None:
        return None
    return d < -thr if side == "low" else d > thr


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


def judge_buried_dry(res, w0, w1, ref):
    dbg = res.get("dbg")
    if not dbg:
        return dict(metric="A ドライ部分ラウドネス", value=None, note="ドライ再レンダーなし")
    bl = [b for b in dbg["A_dry"]["blocks"] if b[1] is not None]
    inwin = [b for b in bl if w0 <= b[0] < w1]
    if not inwin:
        return None
    v = float(np.median([b[1] for b in inwin]))
    allv = [b[1] for b in bl]
    thr_rel = song_pct(allv, 10)
    return dict(metric="A ドライのリード vs オケ+ウェット+ハモリ 損失(phon)", value=v, abs_thr=None, abs_src="—",
                rel_thr=thr_rel, hit_abs=None, hit_rel=v < thr_rel, n_similar=f"{sum(1 for x in allv if x <= v)}/{len(allv)} ブロック")


def judge_estoi(res, w0, w1, ref):
    dbg = res.get("dbg")
    rows = [r for r in (dbg or {}).get("I_estoi", []) if r[2] is not None]
    if not rows:
        return dict(metric="I ESTOI", value=None, note="ドライ再レンダーなし")
    inwin = [r for r in rows if overlaps(r[0], r[1], w0, w1)]
    if not inwin:
        return None
    v = min(r[2] for r in inwin)
    allv = [r[2] for r in rows]
    orig = (res.get("intel_orig") or {})
    thr_abs = orig.get("estoi_p10")
    thr_rel = song_pct(allv, 10)
    return dict(metric="I ESTOI（処理前→マスター、フレーズ）", value=v, abs_thr=thr_abs, abs_src="本家（分離ボーカル→原曲）p10",
                rel_thr=thr_rel, hit_abs=fires(v, thr_abs, "low"), hit_rel=fires(v, thr_rel, "low"),
                n_similar=f"{sum(1 for x in allv if x <= v)}/{len(allv)} フレーズ")


def judge_cer(res, w0, w1, ref):
    asr = res.get("asr")
    rows = [r for r in (asr or {}).get("chunks", []) if r[2] is not None]
    if not rows:
        return dict(metric="I ASR CER", value=None, note="ASR 未実行")
    inwin = [r for r in rows if overlaps(r[0], r[1], w0, w1)]
    if not inwin:
        return None
    r = max(inwin, key=lambda r: r[2])
    allv = [x[2] for x in rows]
    orig = (res.get("intel_orig") or {})
    thr_abs = orig.get("cer_p90")
    thr_rel = song_pct(allv, 90)
    return dict(metric="I ASR 文字誤り率（処理前の書き起こし基準、マスター）", value=r[2], abs_thr=thr_abs,
                abs_src="本家（分離ボーカル→原曲）p90", rel_thr=thr_rel, hit_abs=fires(r[2], thr_abs, "high"),
                hit_rel=fires(r[2], thr_rel, "high"), n_similar=f"{sum(1 for x in allv if x >= r[2])}/{len(allv)} チャンク",
                extra=f"処理前「{r[3][:24]}」/ マスター「{r[4][:24]}」")


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


# Dip audibility: a 1/3-octave dip has to be roughly 3 dB deep or more before listeners detect it in
# broadband material (dips are less audible than peaks; Buecklein 1981, JAES 29(3)). Used as the
# absolute floor for the de-essing traces (a)/(c); the song's own 10th percentile is the relative test.
DIP_AUDIBLE_DB = 3.0


def _q_rows(dbg, name):
    return [q for q in (dbg or {}).get(name, []) if q[3] is not None]


def judge_deess(res, w0, w1, ref, base="Q_dry_raw"):
    """'crushed' sibilance = over-de-essed sound (user, 2026-11-26): traces on the dry lead.
    base Q_dry_raw: processed dry lead vs raw input lead; Q_dry: vs the v7 prepared (after de-ess)."""
    dbg = res.get("dbg")
    rows = _q_rows(dbg, base)
    label = "raw 基準" if base == "Q_dry_raw" else "prepared 基準"
    if not rows:
        return dict(metric=f"Q ディエッサー痕跡 {label}", value=None, note="ドライ再レンダーなし")
    inwin = [q for q in rows if overlaps(q[0], q[1], w0, w1)]
    if not inwin:
        return dict(metric=f"Q ディエッサー痕跡 {label}", value=None, note="窓内に歯擦音なし")
    q = min(inwin, key=lambda q: q[3])
    band = [r[3] for r in rows]
    dip = [r[5] for r in rows if r[5] is not None]
    cen = [r[4] for r in rows if r[4] is not None]
    thr_b, thr_d, thr_c = song_pct(band, 10), song_pct(dip, 10), song_pct(cen, 90)
    hit_abs = (q[3] <= -DIP_AUDIBLE_DB) or (q[5] is not None and q[5] <= -DIP_AUDIBLE_DB)
    hit_rel = (q[3] <= thr_b) or (q[5] is not None and q[5] <= thr_d)
    sim = sum(1 for r in rows if r[3] <= q[3])
    prep = {(r[0], r[1]): r for r in _q_rows(dbg, "Q_prep_raw")}.get((q[0], q[1]))
    extra = (f"@{fmt_time(q[0])} ピーク {q[2]:.0f} Hz, (b)重心低下 {fmt(q[4], 1)}% (曲内 p90 {fmt(thr_c, 1)}), "
             f"(c)ゲインの谷 {fmt(q[5], 1)} dB (曲内 p10 {fmt(thr_d, 1)}), 残差 {fmt(q[6], 1)} dB, LSD {fmt(q[7], 1)} dB")
    if prep and base == "Q_dry_raw":
        extra += f"; うち下処理(prepared−raw) 帯域 {fmt(prep[3], 1)} / 谷 {fmt(prep[5], 1)} dB"
    return dict(metric=f"Q ディエッサー痕跡 {label}: (a)帯域減衰 dB", value=q[3], abs_thr=-DIP_AUDIBLE_DB,
                abs_src="1/3 oct の谷の可聴 ~3 dB (Bücklein 1981)", rel_thr=thr_b, hit_abs=hit_abs, hit_rel=hit_rel,
                n_similar=f"{sim}/{len(rows)} 個", extra=extra)


def judge_consonant(res, w0, w1, ref, kind, direction):
    """Consonant amount on the dry lead (C_dry) if the render has debug stems, else on the bus."""
    dbg = res.get("dbg")
    use_dry = bool(dbg and dbg.get("C_dry"))
    ev = [c for c in (dbg["C_dry"] if use_dry else res.get("C", [])) if c[0] == kind]
    inwin = [c for c in ev if overlaps(c[1], c[2], w0, w1) and c[3] is not None]
    name = "ドライ" if use_dry else "バス"
    if not inwin:
        return dict(metric=f"C {kind} Δ子音/母音({name})", value=None, note="窓内に検出なし")
    allv = [c[3] for c in ev if c[3] is not None]
    pro_abs = (ref["pro"]["consonants"].get(kind) or {}).get("ratio_abs") or {}
    if direction == "crushed":
        c = min(inwin, key=lambda c: c[3])
        thr_rel = song_pct(allv, 10)
        hit_rel, thr_abs, hit_abs, src_abs = fires(c[3], thr_rel, "low"), None, None, "（量ではなく痕跡で判定: Q 行）"
        sim = sum(1 for x in allv if x <= c[3])
    else:
        c = max(inwin, key=lambda c: c[3])
        thr_rel = song_pct(allv, 90)
        hit_rel = fires(c[3], thr_rel, "high")
        # absolute: the processed consonant/vowel ratio itself against the pro distribution (not a delta)
        thr_abs = pro_abs.get("p95")
        hit_abs = fires(c[6], thr_abs, "high")
        src_abs = "比の絶対値 vs プロ p95"
        sim = sum(1 for x in allv if x >= c[3])
    return dict(metric=f"C {kind} Δ子音/母音({name}, dB)", value=c[3], abs_thr=thr_abs, abs_src=src_abs,
                rel_thr=thr_rel, hit_abs=hit_abs, hit_rel=hit_rel, n_similar=f"{sim}/{len(allv)} 個",
                extra=f"@{fmt_time(c[1])} 比の絶対値 {fmt(c[6], 1)} dB, V/I {fmt(c[4], 1)} dB, Δcrest {fmt(c[5], 1)}")


def judge_step(res, w0, w1, ref):
    dbg = res.get("dbg")
    use_dry = bool(dbg and dbg.get("D_dry"))
    d = dbg["D_dry"] if use_dry else res["D"]
    src = "リード+ウェット" if use_dry else "リード推定"
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
    return dict(metric=f"D 段差 far_score(dB, 前後 3 s, {src})", value=v, abs_thr=LEVEL_JND_DB, abs_src="音量 JND 1 dB",
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


def locate_lyric(res, issue, transcriber):
    """Narrow the window to the sung syllable named in the issue (e.g. 真の髄から（「し」）) using
    Whisper word timestamps on the pre vocal. Returns (w0, w1, note) or None if not found."""
    m = re.search(r"「(.+?)」", issue.get("lyric", "").split("（")[-1]) if "（" in issue.get("lyric", "") else None
    if transcriber is None or not m:
        return None
    from _common import load_ch_first  # noqa: F401
    from _v8eval import _cached, _key, render_signals

    from mix_automation.v8.metrics.intelligibility import to_hiragana

    d = Path(res["render_dir"])
    t = float(issue["time_sec"])
    a, b = max(0.0, t - 3.0), t + 5.0
    target = to_hiragana(m.group(1))

    def run():
        sig = render_signals(d)
        pre = true_pre_vocal(d, sig, load_report(d))
        # no initial prompt: Whisper tends to drop a line that repeats the prompt text
        _, words = transcriber.transcribe(pre[int(a * 44100) : int(b * 44100)], 44100, words=True)
        return {"words": [[w.start + a, w.end + a, w.text] for w in words]}

    ws = _cached(f"lyric_{issue['id']}", _key("lyric", issue["id"], d.name, t, "noprompt"), run)["words"]
    w0, w1, _ = window_of(issue)
    ws = [w for w in ws if w[1] >= w0 and w[0] <= w1]
    # concatenated hiragana with a char -> word index map, so multi-syllable targets can span words
    chars, owner = [], []
    for k, w in enumerate(ws):
        h = to_hiragana(w[2])
        chars.append(h)
        owner += [k] * len(h)
    text = "".join(chars)
    i = text.find(target) if target else -1
    if i < 0:
        return None
    k0, k1 = owner[i], owner[i + len(target) - 1]
    words_txt = "".join(w[2] for w in ws[k0 : k1 + 1])
    return (ws[k0][0] - 0.15, ws[k1][1], f"Whisper: 「{words_txt}」 {fmt_time(ws[k0][0])}–{fmt_time(ws[k1][1])}")


def judge_local(res, issue, ref, transcriber=None):
    w0, w1, t_iss = window_of(issue)
    located = locate_lyric(res, issue, transcriber) if issue["category"] == "sibilance" else None
    if located:
        # use the located syllable only if a sibilant was detected there (Whisper can hallucinate)
        ev = [c for c in ((res.get("dbg") or {}).get("C_dry") or res.get("C", [])) if c[0] == "sibilant"]
        if any(overlaps(c[1], c[2], located[0], located[1]) for c in ev):
            w0, w1 = located[0], located[1]
        else:
            located = None
    cat, d = issue["category"], issue["direction"]
    if cat == "buried":
        return [judge_buried(res, w0, w1, ref), judge_buried_dry(res, w0, w1, ref), judge_estoi(res, w0, w1, ref), judge_cer(res, w0, w1, ref)]
    if cat == "muddy":
        return [judge_colour_local(res, w0, w1, t_iss, ref, "bright_rel", "low")]
    loc_note = located[2] if located else "Whisper で特定できず: 時刻窓で最悪の子音"
    if cat == "sibilance" and d == "crushed":
        out = [judge_consonant(res, w0, w1, ref, "sibilant", d), judge_deess(res, w0, w1, ref, "Q_dry_raw"), judge_deess(res, w0, w1, ref, "Q_dry")]
        for o in out:
            if o is not None:
                o["extra"] = f"[{loc_note}] " + (o.get("extra") or o.get("note") or "")
        return out
    if cat == "sibilance":
        out = [judge_consonant(res, w0, w1, ref, "sibilant", d)]
        for o in out:
            if o is not None:
                o["extra"] = f"[{loc_note}] " + (o.get("extra") or o.get("note") or "")
        return out
    if cat == "consonant_attack":
        return [judge_consonant(res, w0, w1, ref, "attack", d)]
    if cat == "step_change":
        return [judge_step(res, w0, w1, ref)]
    if cat == "delay_offbeat":
        return [judge_delay(res, w0, w1, ref)]
    if cat == "fx_dropout":
        return [judge_dropout(res, w0, w1, ref)]
    if cat == "harmony_prominent":
        return [judge_harmony(res, w0, w1, ref, "prom"), judge_harmony(res, w0, w1, ref, "rel_db")]
    return [dict(metric="(対応指標なし)", value=None, note="未対応カテゴリ")]


def judge_global_step(res, issue, ref):
    """'声質がパートごとに変わる': distinct timbre steps (|Δbright| >= JND) of the dry lead + wet vs the
    pre vocal, and how many fall on v7 section boundaries (+-1.5 s)."""
    dbg = res.get("dbg")
    d = dbg["D_dry"] if dbg and dbg.get("D_dry") else res["D"]
    t = np.array(d["t"], dtype=float)
    br = np.array([np.nan if x is None else x for x in d["d_bright"]], dtype=float)
    lv = np.array([np.nan if x is None else x for x in d["d_level"]], dtype=float)
    order = np.argsort(-np.nan_to_num(np.abs(br), nan=-1))
    peaks = []
    for k in order:
        if not np.isfinite(br[k]) or abs(br[k]) < LEVEL_JND:
            break
        if all(abs(t[k] - u) >= 3.0 for u, _ in peaks):
            peaks.append((float(t[k]), float(br[k])))
    bounds = [x["start_sec"] for x in (load_report(Path(res["render_dir"])).get("sections") or [])][1:]
    on_b = sum(1 for u, _ in peaks if any(abs(u - b) <= 1.5 for b in bounds))
    dur_min = res["duration"] / 60.0
    lv_peaks = sum(1 for k in range(len(lv)) if np.isfinite(lv[k]) and abs(lv[k]) >= LEVEL_JND)
    return [dict(metric="D 声質の段差: |Δbright| ≥ JND の箇所", value=float(len(peaks)), abs_thr=1.0,
                 abs_src="1 箇所以上（JND 1 dB）", rel_thr=None, hit_abs=len(peaks) >= 1, hit_rel=None,
                 n_similar=f"{len(peaks) / dur_min:.1f} 箇所/分",
                 extra=f"うち v7 区間境界 ±1.5 s: {on_b}; 時刻 " + ", ".join(f"{fmt_time(u)}({b:+.1f})" for u, b in sorted(peaks)[:8])
                 + f"; 参考: |Δlevel|≥1 dB のフレーム {lv_peaks}")]


def judge_harmony_missing(res, issue, ref):
    dbg = res.get("dbg")
    if not dbg:
        return [dict(metric="H ハモリの有無", value=None, note="ドライ再レンダーなし")]
    v = dbg["E_dry"].get("harmony_bus_rel_db")
    return [dict(metric="H ハモリバス / リード (dB、曲全体)", value=v, abs_thr=None, abs_src="参照なし（本家のハモリは分離できない）",
                 rel_thr=None, hit_abs=None, hit_rel=None, n_similar="—",
                 note="現時点で検出不可（値のみ）。エバは録音ハモリ無し→Nectar 3 生成ハモリが 2mix に入る設計")]


def judge_global(res, issue, ref, song_scalars):
    cat, song = issue["category"], issue["song"]
    if cat == "step_change":
        return judge_global_step(res, issue, ref)
    if cat == "harmony_missing":
        return judge_harmony_missing(res, issue, ref)
    orig = ref.get("original", {}).get(song)
    v7sep = ref.get("v7_separated", {}).get(song)
    out = []
    if cat == "airy_fizzy" and issue["direction"] == "fizzy":
        # Self-mix threshold removed (user 2026-11-26: self mixes are not a ceiling). The processing
        # delta is shown for reference; the judgement is R_airy_abs against the original below.
        fz = res["B"].get("fizz") or {}
        out.append(dict(metric="B fizz: Δ(8-16k − 1-4k) 処理後−処理前 (dB、参考)", value=fz.get("hf_rel_delta"), abs_thr=None,
                        abs_src="（自MIX 基準は廃止）", rel_thr=None, hit_abs=None, hit_rel=None,
                        n_similar="—", extra=f"Δflatness {fmt(fz.get('flat_delta'), 3)}"))
    for rid, _, key, side, label, thr, kind in CATEGORY_RULES.get(cat, []):
        if issue["direction"] == "fizzy" and rid == "R_airy":
            continue
        if orig is None:
            out.append(dict(metric=f"{rid}: {label}", value=None, note="本家音源なし"))
            continue
        d_sep = rel_delta(v7sep, orig, key, kind)
        d_true = rel_delta(song_scalars, orig, key, kind)
        dry = (res.get("dry_scalars") or {})
        d_dry = rel_delta(dry, orig, key, kind) if dry else None
        out.append(dict(metric=f"{rid}: {label}", value=d_sep, abs_thr=(-thr if side == "low" else thr),
                        abs_src="JND（v7 分離版−本家 分離）", rel_thr=None, hit_abs=fires_rel(d_sep, thr, side), hit_rel=None,
                        n_similar="—", extra=f"v7 ステム−本家 {fmt(d_true)}; ドライ再レンダー−本家 {fmt(d_dry)}"))
    if not out:
        out.append(dict(metric="(対応指標なし)", value=None, note="未対応カテゴリ"))
    return out


# --------------------------------------------------------------------------- main


def render_dirs(args) -> list[Path]:
    if args.renders:
        return [Path(p) for p in args.renders]
    out = []
    for d in V7_DIRS.values():
        base = d.split("_auto")[0] + "_auto"
        re = PHASE2_OUT / base
        out.append(re if (re / "debug_stems" / "lead_dry.wav").exists() and (re / "mix_report.json").exists() else ARCHIVE / d)
    return out


def intelligibility_of_original(song: str, dname: str, transcriber) -> dict:
    """ESTOI / ASR CER of the original: separated original vocal (reference) inside the original mix."""
    import numpy as np  # noqa: F811
    from _common import load_ch_first
    from _v8eval import _cached, _fid, _key

    from mix_automation.v8.metrics import intelligibility
    from mix_automation.v8.metrics._core import phrases, to_mono

    stem = dname.split("_auto")[0]
    voc = PROC / "orig_stems" / f"orig_{stem}_vocals.wav"
    rep = load_report(ARCHIVE / dname)
    ia = rep.get("input_assets") or {}
    folder = Path(ia.get("song_folder", ""))
    hits = list(folder.rglob(ia.get("original_file", "__none__"))) if folder.is_dir() and ia.get("original_file") else []
    if not voc.exists() or not hits:
        return {}
    mix_path = hits[0]

    def run_estoi() -> dict:
        v = to_mono(load_ch_first(voc))
        m = to_mono(load_ch_first(mix_path))
        n = min(len(v), len(m))
        segs, _, _ = phrases(v[:n], 44100)
        rows = intelligibility.estoi_segments(v[:n], m[:n], 44100, segs)
        return {"rows": rows}

    es = _cached(f"intel_orig_{stem}", _key("intel_orig", _fid(voc), _fid(mix_path)), run_estoi)
    vals = [r[2] for r in es["rows"] if r[2] is not None]
    out = {"estoi_p10": float(np.percentile(vals, 10)) if vals else None, "estoi_median": float(np.median(vals)) if vals else None}
    if transcriber is not None:
        a = asr_cer(f"orig_{stem}", voc, mix_path, None, transcriber)
        c = [r[2] for r in a["chunks"] if r[2] is not None]
        out |= {"cer_p90": float(np.percentile(c, 90)) if c else None, "cer_median": float(np.median(c)) if c else None}
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("renders", nargs="*")
    ap.add_argument("--write-md", action="store_true")
    ap.add_argument("--compute-only", action="store_true")
    ap.add_argument("--asr", action="store_true", help="also run Whisper large-v3 CER (GPU, slow the first time)")
    args = ap.parse_args()

    transcriber = None
    if args.asr:
        from mix_automation.v8.metrics.intelligibility import Transcriber

        transcriber = Transcriber()
    results = {}
    for d in render_dirs(args):
        res = analyze_render(d)
        res["render_dir"] = str(d)
        res["dbg"] = analyze_debug_render(d)
        dname = next((v for k, v in V7_DIRS.items() if k == res["song"]), None)
        if transcriber is not None:
            from _v8eval import render_signals

            sig = render_signals(d)
            pre = true_pre_vocal(d, sig, load_report(d))
            res["asr"] = asr_cer(f"v7_{d.name}", pre, sig["master"], None, transcriber, ref_id=f"truepre:{sig['pre']}")
        res["intel_orig"] = intelligibility_of_original(res["song"], dname, transcriber) if dname else {}
        results[res["song"]] = res
        print(f"computed {res['song']} ({'debug' if res['dbg'] else 'archive'})", flush=True)
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
        js = judge_global(res, iss, ref, song_sc[iss["song"]]) if iss.get("scope") == "global" else judge_local(res, iss, ref, transcriber)
        for j in js:
            if j is None:
                j = dict(metric="—", value=None, note="窓内にデータなし")
            hit = None
            if j.get("hit_abs") is not None or j.get("hit_rel") is not None:
                hit = bool(j.get("hit_abs")) or bool(j.get("hit_rel"))
            rows.append({"id": iss["id"], "cat": f"{iss['category']}/{iss['direction']}", "when": "全体" if iss.get("scope") == "global" else fmt_time(iss["time_sec"]), **j, "hit": hit})

    # Global rules on every song (verdict) and on negatives: self mixes vs the same original.
    verdict_rows, fa_rows = [], []
    for song in song_sc:
        orig = ref.get("original", {}).get(song)
        if orig is None:
            verdict_rows.append([song, "判定不可", "本家音源なし"])
            continue
        fired = []
        for rid, cat, key, side, label, thr, kind in GLOBAL_RULES:
            d = rel_delta(ref.get("v7_separated", {}).get(song), orig, key, kind)
            if fires_rel(d, thr, side):
                fired.append(f"{rid}({fmt(d)})")
        verdict_rows.append([song, "FAIL" if fired else "PASS", ", ".join(fired) or "—"])
    for rid, cat, key, side, label, thr, kind in GLOBAL_RULES:
        selfs = {s: fires_rel(rel_delta(r, ref.get("original", {}).get(s), key, kind), thr, side) for s, r in ref["self_mix"]["rows"].items()}
        cov = {s: fires_rel(rel_delta(r, ref.get("original", {}).get(s), key, kind), thr, side) for s, r in ref.get("same_song_cover", {}).items()}
        v7 = {s: fires_rel(rel_delta(ref.get("v7_separated", {}).get(s), ref.get("original", {}).get(s), key, kind), thr, side) for s in song_sc}
        def show(dct):
            return " ".join(f"{s[:4]}:{'発火' if v else ('なし' if v is not None else '—')}" for s, v in dct.items())
        fa_rows.append([rid, show(v7), show(cov), show(selfs)])

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
    print_table(["全体ルール", "v7（分離）", "同曲プロカバー（否定例）", "自MIX（参考・判定に使わない）"], fa_rows)

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
        "## 曲ごとの合否（本家との差の全体ルールのどれかが発火したら FAIL）",
        "",
        md_table(["曲", "判定", "発火した全体ルール"], verdict_rows),
        "",
        "## 全体ルールの誤報（否定例）",
        "",
        "全体ルールは本家（原曲を同じ BS-RoFormer で分離）との差を JND と比べる。v7 もマスターを同じモデルで分離した値を使う。"
        "同じ曲のプロカバー（ラブ&デストロイの梓川カバーのみ）で発火したら誤報。自MIX は凌の回答（2026-11-26「納得してない、天井にしないで」）により否定例・閾値の出典から外し、参考に値だけ載せる。"
        "Fall in Love は本家音源が無いので判定不可（fil-g2 も検証できない）。",
        "",
        md_table(["全体ルール", "v7（分離）", "同曲プロカバー（否定例）", "自MIX（参考・判定に使わない）"], fa_rows),
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
