"""v8 Phase 3: which v7 processing stage crushes / over-emphasises a consonant.

Reads the per-stage taps written by `mix-auto mix --export-debug-stems` (`debug_stems/stages/`),
detects consonants once on the chain input, and measures every stage against the stage before it
with the Phase 2 metrics (C: Δ consonant/vowel; Q: de-esser traces — band attenuation, sibilant-band
gain dip, centroid drop). The stage that moves the flagged consonant furthest in the direction of
the user's note is reported as the cause, together with how that stage treats every consonant of
the same kind in the song (systematic or local).

  python scripts/analysis/stage_attribution.py              # all Phase 3 issues, short tables
  python scripts/analysis/stage_attribution.py --covers     # + six-9 / lad-1 phrases vs the human covers

Full per-stage tables go to docs/evaluations/stage_attribution.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, fmt_time, print_table  # noqa: E402
from _v8eval import _cached, _fid, _key  # noqa: E402

from mix_automation.v8.metrics import consonants as C  # noqa: E402
from mix_automation.v8.metrics._core import band_power, db, stft_power, to_mono  # noqa: E402

SR = 44100
OUT3 = ROOT / "data" / "outputs_phase3"
MD = ROOT / "docs" / "evaluations" / "stage_attribution.md"
JND_DB = 1.0  # level JND used throughout Phase 2

LEAD = [
    ("lead_00_raw", "入力"), ("lead_01_rx", "RX 掃除＋ゲート"), ("lead_02_pitch", "ピッチ補正"),
    ("lead_03_rider", "ゲインライダー"), ("lead_04_deess", "RX ディエッサー"), ("lead_05_soothe2_pre", "Soothe2（下処理）"),
    ("lead_06_prepared", "ステージング"), ("lead_11_pq", "Pro-Q（EQ・動的帯域）"), ("lead_12_vca", "VCA コンプ"),
    ("lead_13_la2a", "LA-2A"), ("lead_14_color", "色付け EQ"), ("lead_15_sat", "サチュレーション"),
    ("lead_16_soothe2", "Soothe2（インサート）"), ("lead_17_bark", "子音シェイパー"), ("lead_18_fader", "フェーダー"),
    ("lead_30_multisec", "マルチセクション彫刻"), ("lead_31_gainenv", "区間・距離ゲイン"), ("lead_32_soften", "距離トーン"),
    ("lead_33_inlinefx", "インライン FX"), ("master_lead_bus", "リード最終ゲイン"), ("master_vocal_bus", "バス（ハモリ・空間を加算）"),
    ("master_40_mix", "2mix（オケ加算）"), ("master_41_glue", "バスグルー"), ("master_42_master", "マスター（リミッター）"),
]
AUX = [
    ("{p}20_in", "入力"), ("{p}21_eq", "EQ"), ("{p}22_comp", "コンプ"), ("{p}23_sib", "ハモリ歯擦音ソフナー"),
    ("{p}24_space", "幅・空間"), ("{p}25_level", "音量合わせ"), ("{p}26_pocket", "リード追従"),
    ("master_vocal_bus", "バス（リード等を加算）"), ("master_40_mix", "2mix（オケ加算）"), ("master_41_glue", "バスグルー"),
    ("master_42_master", "マスター（リミッター）"),
]
DETECT_ON = {"lead": "lead_01_rx"}
# stage groups that act together (their per-stage changes add up)
GROUPS = [
    ("コンプ群（Pro-Q・VCA・LA-2A）", ("lead_11_pq", "lead_12_vca", "lead_13_la2a")),
    ("動的高域群（RX ディエッサー・Pro-Q・Soothe2）", ("lead_04_deess", "lead_11_pq", "lead_16_soothe2")),
]


def group_stage(stages: list[dict], names: tuple[str, ...], label: str) -> dict | None:
    sel = [s for s in stages if s["name"] in names]
    if not sel:
        return None
    n = len(sel[0]["d_ratio"])

    def add(vals):
        v = [x for x in vals if x is not None]
        return sum(v) if v else None

    return dict(name="group", label=label, lag=0,
                d_ratio=[add([s["d_ratio"][k] for s in sel]) for k in range(n)],
                q=[tuple(add([s["q"][k][j] for s in sel]) for j in range(3)) for k in range(n)])  # aux chains: their own input

ISSUES = [
    # id, song folder prefix, chain, kind, direction, window (s), user's time (s), note
    ("lad-1", "ラブandデストロイ", "lead", "sibilant", "crushed", (45.6, 47.6), 46.6, "「さ」が潰れる"),
    ("six-7", "シックスフィートアンダー", "lead", "attack", "too_strong", (146.0, 152.0), 149.0, "「タ」のアタックが強い"),
    ("six-9", "シックスフィートアンダー", "lead", "sibilant", "too_strong", (184.0, 186.0), 185.0, "「じゅうどずっと」のサ行が強い"),
    ("fil-1", "Fall_in_Love", "lead", "attack", "too_strong", (151.0, 160.0), 154.0, "アドリブのアタックが強い"),
]
# phrases compared with the human covers (absolute values, same separation model)
COVER_PHRASES = [
    ("six-9", "シックスフィートアンダー", (184.0, 187.5)),
    ("lad-1", "ラブ&デストロイ", (45.6, 47.6)),
]


def render_dir(prefix: str) -> Path:
    cands = sorted(d for d in OUT3.glob(f"{prefix}_auto*") if (d / "debug_stems" / "stages").is_dir())
    if not cands:
        raise SystemExit(f"no Phase 3 render with stage taps for {prefix} under {OUT3}")
    return cands[-1]


def chain_files(rdir: Path, chain: str) -> list[tuple[Path, str]]:
    sd = rdir / "debug_stems" / "stages"
    spec = LEAD if chain == "lead" else [(n.format(p=f"{chain}_"), lab) for n, lab in AUX]
    return [(sd / f"{n}.wav", lab) for n, lab in spec if (sd / f"{n}.wav").exists()]


def load_mono(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR, (p, sr)
    return to_mono(x.T).astype(np.float32)


def lag_of(ref: np.ndarray, x: np.ndarray, max_lag: int = 2048) -> int:
    """Sample lag of x vs ref (positive = x late), from the loudest 20 s of ref."""
    n = min(len(ref), len(x))
    w = 20 * SR
    if n <= w:
        s0 = 0
    else:
        nb = n // SR
        e = (ref[: nb * SR].astype(np.float64) ** 2).reshape(nb, SR).sum(axis=1)  # 1 s blocks
        s0 = int(np.clip(np.argmax(np.convolve(e, np.ones(20), mode="valid")) * SR, 0, n - w))
    a, b = ref[s0 : s0 + w].astype(np.float64), x[s0 : s0 + w].astype(np.float64)
    m = 1 << int(np.ceil(np.log2(2 * len(a))))
    cc = np.fft.irfft(np.fft.rfft(b, m) * np.conj(np.fft.rfft(a, m)), m)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    return int(np.argmax(cc)) - max_lag


def shift(x: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return x
    return np.concatenate([x[lag:], np.zeros(lag, x.dtype)]) if lag > 0 else np.concatenate([np.zeros(-lag, x.dtype), x[:lag]])


def chain_measures(rdir: Path, chain: str) -> dict:
    files = chain_files(rdir, chain)
    det = DETECT_ON.get(chain, f"{chain}_20_in")
    det_p = rdir / "debug_stems" / "stages" / f"{det}.wav"

    def run() -> dict:
        x_det = load_mono(det_p)
        events = C.detect_consonants(x_det, SR)
        ev = [dict(kind=e.kind, start=e.start, end=e.end, vs=e.vowel_start, ve=e.vowel_end) for e in events]
        stages, prev, prev_lab = [], None, None
        for p, lab in files:
            x = load_mono(p)
            lag = 0
            if prev is not None:
                lag = lag_of(prev, x)
                x = shift(x, lag)
                n = min(len(prev), len(x))
                cm = C.measure_consonants(events, x[:n], prev[:n], None, SR)
                dr = [m.delta_ratio_db for m in cm]
                tr = {(t.event.start, t.event.end): t for t in C.deess_traces(events, x[:n], prev[:n], SR)}
                q = [(tr[(e.start, e.end)].band_att_db, tr[(e.start, e.end)].gain_dip_db, tr[(e.start, e.end)].centroid_drop_pct)
                     if (e.start, e.end) in tr else (None, None, None) for e in events]
                stages.append(dict(name=p.stem, label=lab, from_label=prev_lab, lag=lag, d_ratio=dr, q=q))
            prev, prev_lab = x, lab
        return dict(events=ev, stages=stages)

    key = _key("stage_attr", 2, chain, *(_fid(p) for p, _ in files), _fid(det_p))
    return _cached(f"stage_attr_{rdir.name}_{chain}", key, run)


def _f(v, nd=1) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{nd}f}"


def pick_event(ev: list[dict], stages: list[dict], kind: str, direction: str, win, t_user) -> int | None:
    """Event in the window whose total processing change (all stages up to the bus) is most in the
    note's direction; ties to the user's time."""
    idx = [i for i, e in enumerate(ev) if e["kind"] == kind and e["start"] < win[1] and e["end"] > win[0]]
    if not idx:
        return None
    proc = [s for s in stages if not s["name"].startswith("master_")]
    tot = {i: sum((s["d_ratio"][i] or 0.0) for s in proc) for i in idx}
    sgn = -1.0 if direction == "crushed" else 1.0
    return max(idx, key=lambda i: (sgn * tot[i], -abs(ev[i]["start"] - t_user)))


def stage_score(s: dict, i: int, direction: str, kind: str) -> float:
    """Move in the note's direction (dB, positive = towards the complaint)."""
    dr = s["d_ratio"][i]
    if direction == "crushed":
        cands = [-(dr or 0.0)]
        ba, dip, _ = s["q"][i]
        cands += [-(ba or 0.0), -(dip or 0.0)]
        return max(cands)
    return dr or 0.0


def systematic(stage: dict, ev: list[dict], kind: str, direction: str, i_target: int) -> dict:
    idx = [i for i, e in enumerate(ev) if e["kind"] == kind]
    vals = np.array([stage_score(stage, i, direction, kind) for i in idx], dtype=float)
    vals = vals[np.isfinite(vals)]
    v = stage_score(stage, i_target, direction, kind)
    return dict(n=len(vals), median=float(np.median(vals)), p10=float(np.percentile(vals, 10)), p90=float(np.percentile(vals, 90)),
                share=float(np.mean(vals >= JND_DB)), rank=float(np.mean(vals <= v) * 100))


def analyse(md: list[str]) -> list[list[object]]:
    summary = []
    for iid, prefix, chain, kind, direction, win, t_user, note in ISSUES:
        rdir = render_dir(prefix)
        res = chain_measures(rdir, chain)
        ev, stages = res["events"], res["stages"]
        i = pick_event(ev, stages, kind, direction, win, t_user)
        md.append(f"\n## {iid} {note}（{rdir.name} / {chain}）\n")
        if i is None:
            md.append(f"窓 {fmt_time(win[0])}–{fmt_time(win[1])} に {kind} の検出なし\n")
            summary.append([iid, "検出なし", "", "", "", ""])
            continue
        e = ev[i]
        md.append(f"対象: {kind} @{fmt_time(e['start'])}（{e['end'] - e['start']:.2f} s）。値は前の段からの変化（dB）。"
                  "Q は歯擦音のみ: 帯域減衰（母音の変化を引いた値）、ゲインの谷、重心低下 %。\n")
        md.append("| 段 | ラグ | Δ子音/母音 | 帯域減衰 | ゲインの谷 | 重心低下 % | 指摘の向きの量 |\n|---|---|---|---|---|---|---|")
        for s in stages:
            ba, dip, cd = s["q"][i]
            md.append(f"| {s['label']} | {s['lag']} | {_f(s['d_ratio'][i])} | {_f(ba)} | {_f(dip)} | {_f(cd)} | {_f(stage_score(s, i, direction, kind))} |")
        proc = [s for s in stages if not s["name"].startswith("master_")]
        cause = max(proc, key=lambda s: stage_score(s, i, direction, kind))
        allst = max(stages, key=lambda s: stage_score(s, i, direction, kind))
        tot = sum((s["d_ratio"][i] or 0.0) for s in proc)
        sysd = systematic(cause, ev, kind, direction, i)
        md.append(f"\n原因の段: **{cause['label']}**（{_f(stage_score(cause, i, direction, kind))} dB）。"
                  f"バス・マスター込みの最大: {allst['label']}（{_f(stage_score(allst, i, direction, kind))}）。"
                  f"処理全体の Δ子音/母音 {_f(tot)} dB。\n")
        md.append(f"曲全体（{kind} {sysd['n']} 個）でのこの段の指摘向きの量: 中央値 {_f(sysd['median'])}、p10 {_f(sysd['p10'])}、"
                  f"p90 {_f(sysd['p90'])}、JND 以上 {sysd['share'] * 100:.0f}%、対象の順位 {sysd['rank']:.0f} パーセンタイル\n")
        gtxt = []
        for glab, gnames in GROUPS:
            g = group_stage(stages, gnames, glab)
            if g is None:
                continue
            gs = systematic(g, ev, kind, direction, i)
            gv = stage_score(g, i, direction, kind)
            md.append(f"{glab}: 対象 {_f(gv)} dB、曲全体 中央値 {_f(gs['median'])}、JND 以上 {gs['share'] * 100:.0f}%、"
                      f"対象の順位 {gs['rank']:.0f}\n")
            gtxt.append(f"{glab.split('（')[0]} {_f(gv)}（中央 {_f(gs['median'])}, 順位 {gs['rank']:.0f}）")
        summary.append([iid, f"@{fmt_time(e['start'])}", cause["label"], _f(stage_score(cause, i, direction, kind)), _f(tot),
                        f"中央 {_f(sysd['median'])} / JND超 {sysd['share'] * 100:.0f}% / 順位 {sysd['rank']:.0f}", "; ".join(gtxt)])
    return summary


def song_wide(md: list[str]) -> None:
    """Median change per stage over all consonants (both kinds), per render/chain."""
    md.append("\n## 段ごとの曲全体の傾向（全子音の中央値、dB）\n")
    done = set()
    for _, prefix, chain, *_ in ISSUES:
        if (prefix, chain) in done:
            continue
        done.add((prefix, chain))
        rdir = render_dir(prefix)
        res = chain_measures(rdir, chain)
        ev = res["events"]
        md.append(f"\n### {rdir.name} / {chain}\n\n| 段 | 歯擦音 Δ子音/母音 | 歯擦音 帯域減衰 | 歯擦音 ゲインの谷 | アタック Δ子音/母音 |\n|---|---|---|---|---|")
        for s in res["stages"]:
            def med(vals):
                a = np.array([v for v in vals if v is not None], dtype=float)
                return float(np.median(a)) if a.size else None
            sib = [k for k, e in enumerate(ev) if e["kind"] == "sibilant"]
            atk = [k for k, e in enumerate(ev) if e["kind"] == "attack"]
            md.append(f"| {s['label']} | {_f(med([s['d_ratio'][k] for k in sib]))} | {_f(med([s['q'][k][0] for k in sib]))} | "
                      f"{_f(med([s['q'][k][1] for k in sib]))} | {_f(med([s['d_ratio'][k] for k in atk]))} |")


def phrase_covers(md: list[str], iid: str, song: str, phrase: tuple[float, float]) -> list[list[object]]:
    """Absolute sibilant/vowel ratio and sibilant centroid of a phrase: v7 (separated master, same
    model as the covers) vs the usable human covers, all on the v7 time axis."""
    import json

    cs = json.loads((ROOT / "docs" / "evaluations" / "cover_stats.json").read_text(encoding="utf-8"))
    from cover_refs import ALIGNED, v7_sep_paths

    sigs = [("v7（マスター分離）", v7_sep_paths(song)[0], "")]
    for slug, idn in cs["identity"].items():
        if idn["song"] == song and idn["kind"] == "cover" and idn["status"] != "exclude":
            sigs.append((idn["name"], ALIGNED / f"{slug}_vocals.wav", idn.get("why") or ""))
    t0, t1 = phrase

    def one(p: Path) -> dict:
        x = load_mono(p)
        s0, s1 = int((t0 - 1.0) * SR), int((t1 + 1.0) * SR)
        seg = x[s0:s1]
        evs = [e for e in C.detect_consonants(seg, SR) if e.kind == "sibilant" and e.start >= 1.0 and e.end <= 1.0 + (t1 - t0)]
        ms = C.measure_consonants(evs, seg, None, None, SR)
        f, t, pw = stft_power(seg, SR)
        cents = []
        for e in evs:
            fr = (t >= e.start) & (t < e.end)
            sel = (f >= 3000.0) & (f < 16000.0)
            pm = pw[sel][:, fr].mean(axis=1)
            cents.append(float((f[sel] * pm).sum() / (pm.sum() + 1e-20)))
        win = (t >= 1.0) & (t < 1.0 + (t1 - t0))
        hf = band_power(pw[:, win], f, 4000.0, 12000.0).sum()
        lf = band_power(pw[:, win], f, 200.0, 4000.0).sum()
        # whole-song context: median absolute ratio of every sibilant
        allm = C.measure_consonants([e for e in C.detect_consonants(x, SR) if e.kind == "sibilant"], x, None, None, SR)
        return dict(n=len(ms), cents=cents, ratios=[m.ratio_proc_db for m in ms], times=[t0 - 1.0 + m.event.start for m in ms],
                    hf_lf=float(db(hf / (lf + 1e-20))), song_med=float(np.median([m.ratio_proc_db for m in allm])) if allm else None)

    rows = []
    md.append(f"\n## {iid}: {fmt_time(t0)}–{fmt_time(t1)} のフレーズ、歯擦音／母音比の絶対値（カバーと同じ分離モデル）\n")
    md.append("| 音源 | 歯擦音の数 | 比の最大 | 比の平均 | 歯擦音の重心 Hz | 4–12k / 0.2–4k | 曲全体の比の中央値 | 注記 |\n|---|---|---|---|---|---|---|---|")
    for name, p, why in sigs:
        r = _cached(f"cover_phrase_{iid}_{name}", _key("cover_phrase", 3, iid, _fid(p), t0, t1), lambda p=p: one(p))
        mx = max(r["ratios"]) if r["ratios"] else None
        mean = float(np.mean(r["ratios"])) if r["ratios"] else None
        cen = f"{np.mean(r['cents']):.0f}" if r["cents"] else "—"
        md.append(f"| {name} | {r['n']} | {_f(mx)} | {_f(mean)} | {cen} | {_f(r['hf_lf'])} | {_f(r['song_med'])} | {why} |")
        rows.append([iid, name, r["n"], _f(mx), _f(mean), cen, _f(r["hf_lf"]), _f(r["song_med"]), why])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", action="store_true", help="also compare the six-9 / lad-1 phrases with the human covers")
    a = ap.parse_args()
    md = ["# v8 Phase 3: 子音の変化の処理段への帰属（stage_attribution.py が生成）\n",
          "各段は直前の段との比較。正の「指摘の向きの量」= 指摘（潰れ／強すぎ）の方向へ動かした量。"
          "バス・2mix・マスター段は混ざった信号での比較（他の音の加算・時間変化するゲインを含む）。"
          "混ざった段の Q（帯域減衰・ゲインの谷）は、オケ等が歯擦音帯域を埋めるため処理の痕跡としては読めない（参考値）。"
          "原因の段は処理段（バス以前）からだけ選ぶ。\n"]
    summary = analyse(md)
    print_table(["指摘", "子音", "原因の段", "量 dB", "処理全体 Δ", "その段の曲全体（中央値 / JND超 / 対象順位）", "群（対象 / 曲全体の中央 / 順位）"], summary)
    song_wide(md)
    if a.covers:
        rows = [r for iid, song, ph in COVER_PHRASES for r in phrase_covers(md, iid, song, ph)]
        print()
        print_table(["指摘", "音源", "歯擦音", "比 最大", "比 平均", "重心 Hz", "HF/LF", "曲全体 中央", "注記"], rows)
    MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n→ {MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
