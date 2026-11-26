"""Reference distributions for the v8 metrics (Phase 2).

Sources (all existing data; no new separation needed):
- Pro covers: BS-RoFormer stems already in data/processed/roformer_stems and reference_stems
  (vocals + inst per song). Consonant measures and the (slow) blind RT60 of the backing on a fixed subset
  (every k-th song, ~20 songs).
- User's own mixes: シックスフィートアンダー, ラブ&デストロイ (separated stems in data/processed/self_mix_sep;
  シックスの 自mixオケなし.wav is only a 26 s excerpt, ラブ&デストロイ の 自mix is 66 s). The v7 prepared vocal of the same song,
  time-aligned, serves as the pre-processing vocal (same raw takes), so processing deltas exist.
- v7 masters separated with the same BS-RoFormer (data/processed/v8_v7_master_sep): checks the bias
  between true stems and separated stems.

Writes docs/evaluations/reference_stats.json (percentiles + per-song rows; small, committed).

    .venv\\Scripts\\python.exe scripts\\analysis\\build_reference_stats.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_ch_first, print_table  # noqa: E402
from _v8eval import analyze_stem_pair, render_signals  # noqa: E402

from mix_automation.v8.metrics._core import to_mono  # noqa: E402

PROC = ROOT / "data" / "processed"
ARCHIVE = ROOT / "data" / "archive" / "gemini_v7_20261126"
OUT = ROOT / "docs" / "evaluations" / "reference_stats.json"
PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)
CONSONANT_SUBSET = 20

V7_DIRS = {
    "Fall in Love": "Fall_in_Love_auto",
    "エバ": "エバ_auto",
    "シックスフィートアンダー": "シックスフィートアンダー_auto",
    "ラブ&デストロイ": "ラブandデストロイ_auto_20261126-135529",
}


def pro_pairs() -> list[tuple[str, Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    for folder in ("roformer_stems", "reference_stems"):
        for v in sorted((PROC / folder).glob("*_vocals.wav")):
            slug = v.name[: -len("_vocals.wav")]
            i = v.with_name(f"{slug}_inst.wav")
            if i.exists() and slug not in pairs:
                pairs[slug] = (v, i)
    return [(k, *v) for k, v in sorted(pairs.items())]


def scalars(r: dict) -> dict[str, float | None]:
    b, e, a = r["B"], r["E"], r["A"]
    vb = b["vocal_bands_db"]
    hf_abs = None
    if vb and all(x is not None for x in vb[2:8]):
        hf_abs = float(np.mean(vb[6:8]) - np.mean(vb[3:6]))
    ratio = None
    if e.get("decay_t60") and e.get("inst_rt60_blind"):
        ratio = e["decay_t60"] / e["inst_rt60_blind"]
    return {
        "loss_median": a["loss_median"],
        "vi1k4k_median": a["vi1k4k_median"],
        "body_rel": b["body_rel"],
        "air_rel": b["air_rel"],
        "bright_rel": b["bright_rel"],
        "tilt": b["tilt"],
        "hf_rel_abs": hf_abs,
        "decay_t60": e["decay_t60"],
        "tail_ratio": e["tail_ratio"],
        "inst_rt60_blind": e["inst_rt60_blind"],
        "decay_over_inst_rt": ratio,
        "side_mid": e["side_mid"],
    }


def pct(values: list[float]) -> dict[str, float] | None:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return {f"p{p}": round(float(np.percentile(v, p)), 3) for p in PCTS} | {"n": int(v.size)}


def consonant_pool(rows: list[list], kind: str, col: int) -> list[float]:
    return [c[col] for c in rows if c[0] == kind and c[col] is not None]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N pro songs (debug)")
    args = ap.parse_args()

    pairs = pro_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    step = max(1, len(pairs) // CONSONANT_SUBSET)
    cons_set = {p[0] for p in pairs[::step][:CONSONANT_SUBSET]}

    pro_rows, pro_phr, pro_cons = {}, [], []
    for k, (slug, v, i) in enumerate(pairs):
        r = analyze_stem_pair(f"pro_{slug}", v, i, with_consonants=slug in cons_set, with_inst_rt=slug in cons_set)
        pro_rows[slug] = scalars(r)
        pro_phr += [b[1] for b in r["A"]["blocks"] if b[1] is not None]
        if "C" in r:
            pro_cons += r["C"]
        print(f"[{k + 1}/{len(pairs)}] {slug}", flush=True)

    # User's own mixes + v7 separated masters (pre = aligned v7 prepared vocal of the same song).
    self_rows, self_cons, self_full = {}, [], {}
    sm_sep = PROC / "self_mix_sep"
    # 自mixオケなし.wav of シックス is a 26 s excerpt only, so both self mixes use the separated stems.
    selfmix_sources = {}
    for song, slug in (("シックスフィートアンダー", "シックスフィートアンダー"), ("ラブ&デストロイ", "ラブandデストロイ")):
        v, i = sm_sep / f"selfmix_{slug}_vocals.wav", sm_sep / f"selfmix_{slug}_inst.wav"
        if v.exists() and i.exists():
            selfmix_sources[song] = (v, i)

    def pre_of(song: str) -> tuple[np.ndarray | None, str]:
        d = ARCHIVE / V7_DIRS.get(song, "")
        s = render_signals(d) if d.is_dir() else {}
        p = s.get("pre")
        return (to_mono(load_ch_first(p)), str(p)) if p else (None, "")

    for song, (v, i) in selfmix_sources.items():
        pre, pid = pre_of(song)
        r = analyze_stem_pair(f"self_{song}", v, i, pre=pre, with_consonants=True, pre_id=pid)
        self_rows[song] = scalars(r)
        self_full[song] = {"fizz": r["B"].get("fizz")}
        self_cons += r.get("C", [])
        print(f"self-mix {song}", flush=True)

    v7sep_rows = {}
    for song, dname in V7_DIRS.items():
        sd = PROC / "v8_v7_master_sep"
        v, i = sd / f"v7master_{dname}_vocals.wav", sd / f"v7master_{dname}_inst.wav"
        if v.exists() and i.exists():
            r = analyze_stem_pair(f"v7sep_{dname}", v, i)
            v7sep_rows[song] = scalars(r)

    keys = list(next(iter(pro_rows.values())).keys()) if pro_rows else []
    stats = {
        "schema_version": 1,
        "description": "Phase 2 reference distributions (scripts/analysis/build_reference_stats.py)",
        "pro": {
            "n_songs": len(pro_rows),
            "song_level": {k: pct([r[k] for r in pro_rows.values()]) for k in keys},
            "block_masking_loss": pct(pro_phr),
            "consonant_subset_songs": sorted(cons_set),
            "consonants": {
                kind: {"ratio_abs": pct(consonant_pool(pro_cons, kind, 6)), "vi": pct(consonant_pool(pro_cons, kind, 4))}
                for kind in ("sibilant", "attack")
            },
            "rows": pro_rows,
        },
        "self_mix": {
            "rows": self_rows,
            "fizz": self_full,
            "consonants": {
                kind: {
                    "delta_ratio": pct(consonant_pool(self_cons, kind, 3)),
                    "delta_crest": pct(consonant_pool(self_cons, kind, 5)),
                    "vi": pct(consonant_pool(self_cons, kind, 4)),
                    "ratio_abs": pct(consonant_pool(self_cons, kind, 6)),
                }
                for kind in ("sibilant", "attack")
            },
        },
        "v7_separated": v7sep_rows,
    }
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    sl = stats["pro"]["song_level"]
    print_table(
        ["metric", "p5", "p50", "p95", "n"],
        [[k, sl[k]["p5"], sl[k]["p50"], sl[k]["p95"], sl[k]["n"]] for k in keys if sl.get(k)],
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
