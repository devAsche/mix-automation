"""Phase 2c judgement against human pro covers of the same song (docs/evaluations/cover_stats.json,
built by cover_refs.py stats).

Rule (2026-11-26 review): a global rule fires when the v7 value (master separated with the same
BS-RoFormer) lies outside the range (min..max) of the human references of that song, at least one JND
beyond the nearer edge, on the side the issue names. JNDs are those of evaluate_issues.GLOBAL_RULES.
References: human covers only (status use/flag); for Fall in Love the original (a human singer) is added.
The vocaloid originals (エバ, シックス, ラブ&デストロイ) are shown for Side/Mid only.
False alarms: leave-one-out - each cover judged against the range of the remaining references.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COVER_STATS = Path(__file__).resolve().parents[2] / "docs" / "evaluations" / "cover_stats.json"
HUMAN_ORIGINAL = {"Fall in Love"}  # original sung by a human -> joins the references
MIN_REFS = 3
PHRASE_JND = 1.0  # loudness level JND ~1 phon (same source as LEVEL_JND, Zwicker & Fastl sec. 7.1)
# fil-g2 "距離感は OK": rules that read distance (level/masking and reverberation cues).
DISTANCE_RULES = ("R_buried", "R_sits_loss", "R_sits_dry", "R_space_tail", "R_space_decay")


def load() -> dict | None:
    return json.loads(COVER_STATS.read_text(encoding="utf-8")) if COVER_STATS.exists() else None


def refs_of(cs: dict, song: str, exclude: str | None = None) -> dict[str, dict]:
    s = cs["songs"].get(song) or {}
    out = {}
    for slug, r in (s.get("refs") or {}).items():
        if slug == exclude or r["status"] == "exclude":
            continue
        if r["kind"] == "cover" or (r["kind"] == "original" and song in HUMAN_ORIGINAL):
            out[slug] = r
    return out


def beyond(v: float | None, vals: list[float], side: str, thr: float, kind: str) -> tuple[bool | None, float | None]:
    """(fires, distance beyond the nearer edge; positive = outside on `side`)."""
    vals = [x for x in vals if x is not None]
    if v is None or len(vals) < MIN_REFS:
        return None, None
    lo, hi = min(vals), max(vals)
    if kind == "ratio":
        d = (lo - v) / lo if side == "low" else (v - hi) / hi
    else:
        d = lo - v if side == "low" else v - hi
    return d >= thr, d


def judge_rule(cs: dict, song: str, rule: tuple, value: float | None = None, exclude: str | None = None) -> dict:
    rid, _, key, side, label, thr, kind = rule
    refs = refs_of(cs, song, exclude)
    vals = [r["scalars"].get(key) for r in refs.values()]
    v = cs["songs"][song]["v7_sep"].get(key) if value is None else value
    fire, d = beyond(v, vals, side, thr, kind)
    vv = [x for x in vals if x is not None]
    return {"rid": rid, "value": v, "lo": min(vv) if vv else None, "hi": max(vv) if vv else None, "n": len(vv),
            "dist": d, "fires": fire}


def loo(cs: dict, rules: list) -> dict:
    """{rid: [fired, trials]} over every cover of every song, judged against the other references."""
    out = {r[0]: [0, 0] for r in rules}
    per_song: dict[str, list[int]] = {}
    for song in cs["songs"]:
        for slug, r in refs_of(cs, song).items():
            if r["kind"] != "cover":
                continue
            for rule in rules:
                j = judge_rule(cs, song, rule, value=r["scalars"].get(rule[2]), exclude=slug)
                if j["fires"] is None:
                    continue
                out[rule[0]][1] += 1
                out[rule[0]][0] += int(j["fires"])
                ps = per_song.setdefault(song, [0, 0])
                ps[0] += int(j["fires"])
                ps[1] += 1
    return {"rules": out, "songs": per_song}


def _ok(x) -> bool:
    return x is not None and np.isfinite(x)


def phrase_compare(cs: dict, song: str, windows: dict[str, tuple[float, float]], split_at: dict[str, float]) -> dict:
    """Per v7 phrase: v7 masking loss vs the cover range (absolute) and local contrast (phrase minus the
    mean of its neighbours) vs the cover range. windows: issue id -> (t0, t1). split_at: issue id ->
    time of an 'after is worse than before' statement (phrase right after minus right before)."""
    s = cs["songs"][song]
    segs = s["v7_phrases"]
    v7 = s["v7_phrase_loss"]
    covers = {k: r["phrase_loss"] for k, r in refs_of(cs, song).items()}
    n = len(segs)

    def contrast(arr: list) -> list:
        out = []
        for i in range(n):
            nb = [arr[j] for j in (i - 1, i + 1) if 0 <= j < n and _ok(arr[j])]
            out.append(arr[i] - float(np.mean(nb)) if _ok(arr[i]) and nb else None)
        return out

    v7c = contrast(v7)
    cc = {k: contrast(a) for k, a in covers.items()}
    rows = []
    for i, (a, b) in enumerate(segs):
        f_abs, d_abs = beyond(v7[i], [c[i] for c in covers.values()], "low", PHRASE_JND, "db")
        f_loc, d_loc = beyond(v7c[i], [c[i] for c in cc.values()], "low", PHRASE_JND, "db")
        cv = [c[i] for c in covers.values() if _ok(c[i])]
        rows.append({"i": i, "start": a, "end": b, "v7": v7[i], "cov_lo": min(cv) if cv else None, "cov_hi": max(cv) if cv else None,
                     "n": len(cv), "abs": f_abs, "d_abs": d_abs, "v7_contrast": v7c[i], "loc": f_loc, "d_loc": d_loc})
    res = {"rows": rows, "issues": {}, "split": {}}
    in_any = set()
    for iid, (t0, t1) in windows.items():
        sel = [r for r in rows if r["end"] > t0 and r["start"] < t1]
        in_any |= {r["i"] for r in sel}
        res["issues"][iid] = {"phrases": sel, "hit_abs": any(r["abs"] for r in sel), "hit_loc": any(r["loc"] for r in sel)}
    others = [r for r in rows if r["i"] not in in_any]
    res["similar_abs"] = sum(1 for r in others if r["abs"])
    res["similar_loc"] = sum(1 for r in others if r["loc"])
    res["n_phrases"] = n
    for iid, t in split_at.items():
        before = [r for r in rows if r["end"] <= t + 0.5]
        after = [r for r in rows if r["start"] >= t - 0.5 and r not in before]
        if not before or not after:
            continue
        ib, ia = before[-1]["i"], after[0]["i"]

        def diff(arr: list) -> float | None:
            return arr[ia] - arr[ib] if _ok(arr[ia]) and _ok(arr[ib]) else None

        dv = diff(v7)
        dc = [diff(a) for a in covers.values()]
        f, d = beyond(dv, dc, "low", PHRASE_JND, "db")
        dcv = [x for x in dc if x is not None]
        res["split"][iid] = {"t_before": segs[ib], "t_after": segs[ia], "v7": dv, "cov_lo": min(dcv) if dcv else None,
                             "cov_hi": max(dcv) if dcv else None, "n": len(dcv), "fires": f, "dist": d}
    return res


def step_negatives(cs: dict, song: str, jnd: float = 1.0) -> list[dict]:
    """At each boundary under test: v7 (separated) raw Δlevel / Δbright vs the covers' range. The v7 step
    is 'v7-specific' only if it lies outside the cover range by >= JND on its own side."""
    s = cs["songs"][song]
    st = s.get("steps") or {}
    v7 = st.get("v7") or []
    out = []
    for k, (name, t) in enumerate(s.get("step_times", {}).items()):
        row = {"name": name, "t": t}
        for feat in ("d_level", "d_bright"):
            v = v7[k][feat] if k < len(v7) else None
            cv = [st[c][k][feat] for c in st if c != "v7" and k < len(st[c]) and st[c][k][feat] is not None]
            side = "low" if (v or 0) < 0 else "high"
            f, d = beyond(v, cv, side, jnd, "db")
            same = sum(1 for x in cv if v is not None and np.sign(x) == np.sign(v) and abs(x) >= jnd)
            row[feat] = {"v7": v, "lo": min(cv) if cv else None, "hi": max(cv) if cv else None, "n": len(cv),
                         "v7_specific": f, "dist": d, "covers_same_sign_jnd": same}
        out.append(row)
    return out
