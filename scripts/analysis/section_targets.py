"""R5b: per-section-type vocal-minus-backing targets from the pro covers (R2 pairs), main-only frames.

    python scripts/analysis/section_targets.py                    # all R2 songs, table + JSON
    python scripts/analysis/section_targets.py --songs エバ --render data/outputs_v8r5b/render/エバ_v8

kVI = K-weighted vocal/backing energy ratio (dB) over the frames of a section where the user's lead is
active and every non-lead stem (double, harmonies, octaves, choir) is inactive (P.56, 512-sample frames),
so the pro target_vocal (which contains the pro's harmonies) is compared on lead-only material.
Pro frames: the same original-time frames mapped to cover time with match.orig_to_cover.
Sections: the v7 render's structure analysis (data/outputs/<song>_auto/mix_report.json), types
Chorus -> chorus, Verse_PreChorus -> verse (A and B melo are not told apart), Quiet_Bridge -> quiet.
Per type: energy-pooled over the type's sections per cover, then the median over covers.
rel = type - verse. Pooled rel = median over songs. --render adds the same numbers for a v8 render
(lead_dry [+ correction EQ not included] + all wet_* vs inst_ducked).

Outputs a short table and data/processed/v8r_sections/section_targets_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SR, print_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "v8r_sections" / "section_targets_analysis.json"
V7_DIRS = {"エバ": "エバ_auto", "ラブ&デストロイ": "ラブandデストロイ_auto", "Fall in Love": "Fall_in_Love_auto"}
R5_RENDERS = {"エバ": "エバ_v8", "ラブ&デストロイ": "ラブandデストロイ_v8"}
FRAME = 512
TYPES = ("verse", "chorus", "quiet")
NONLEAD = ("double_stereo", "sub_octave_stereo", "super_octave_stereo", "harmonies_stereo_bus",
           "main_harmonies_stereo_bus", "choir_stereo_bus")


def sec_type(label: str) -> str | None:
    role = label.split("_", 1)[-1]
    return {"Chorus": "chorus", "Verse_PreChorus": "verse", "Quiet_Bridge_Verse": "quiet"}.get(role)


def _load(path: Path) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    return a.T


def frame_k_energy(x: np.ndarray) -> np.ndarray:
    """Sum over channels of the K-weighted energy per FRAME samples."""
    from mix_automation.v8.dsp import k_weight

    x = np.atleast_2d(x)
    e = 0.0
    for ch in x:
        k = k_weight(ch.astype(np.float64), SR)
        n = len(k) // FRAME
        e = e + np.sum(k[: n * FRAME].reshape(n, FRAME) ** 2, axis=1)
    return np.asarray(e)


def main_only_mask(song: str) -> tuple[np.ndarray, np.ndarray]:
    """(lead active & non-lead inactive, lead active) on the FRAME grid of the user's original timeline."""
    from mix_automation.io.audio_loader import consolidate_vocal_stem_folder, load_audio
    from mix_automation.pipeline import discover_song_assets
    from mix_automation.v8 import chain
    from mix_automation.v8.metrics._core import HOP, p56_activity

    assert HOP == FRAME
    assets = discover_song_assets(ROOT / "data" / "raw" / song)
    stems = consolidate_vocal_stem_folder(list(assets["stems"]), target_sr=SR)
    inst, _ = load_audio(Path(assets["inst"]), target_sr=SR, mono=False)
    n = np.atleast_2d(inst).shape[1]
    lead = chain.fit_len(np.asarray(stems.lead_mono, np.float32), n)
    nonlead = np.zeros(n, np.float32)
    for nm in NONLEAD:
        a = getattr(stems, nm, None)
        if a is not None:
            nonlead += np.abs(chain.fit_len(np.asarray(a, np.float32).mean(0), n))
    la, _, _ = p56_activity(lead, SR)
    if np.any(nonlead > 0):
        na, _, _ = p56_activity(nonlead, SR)
    else:
        na = np.zeros_like(la)
    m = min(len(la), len(na))
    return la[:m] & ~na[:m], la[:m]


def sections(song: str) -> list[tuple[str, str, float, float]]:
    rep = json.loads((ROOT / "data" / "outputs" / V7_DIRS[song] / "mix_report.json").read_text(encoding="utf-8"))
    out = []
    for s in rep["sections"]:
        t = sec_type(s["label"])
        if t is not None:
            out.append((s["label"], t, float(s["start_sec"]), float(s["end_sec"])))
    return out


def per_section(ev: np.ndarray, ei: np.ndarray, frames_of, secs, mask: np.ndarray) -> list[dict]:
    """frames_of(j_orig array) -> frame indices into ev/ei (identity for the user's timeline)."""
    rows = []
    for label, typ, t0, t1 in secs:
        j = np.arange(int(t0 * SR / FRAME), min(int(t1 * SR / FRAME), len(mask)))
        j = j[mask[j]]
        c = frames_of(j)
        ok = (c >= 0) & (c < min(len(ev), len(ei)))
        c = c[ok]
        sv, si = float(np.sum(ev[c])), float(np.sum(ei[c]))
        rows.append({"label": label, "type": typ, "t0": t0, "t1": t1, "frames_s": len(c) * FRAME / SR,
                     "sv": sv, "si": si, "kvi": 10 * np.log10(sv / si) if sv > 0 and si > 0 and len(c) * FRAME >= SR else None})
    return rows


def by_type(rows: list[dict]) -> dict[str, float | None]:
    out = {}
    for t in TYPES:
        sv = sum(r["sv"] for r in rows if r["type"] == t and r["kvi"] is not None)
        si = sum(r["si"] for r in rows if r["type"] == t and r["kvi"] is not None)
        out[t] = 10 * np.log10(sv / si) if sv > 0 and si > 0 else None
    return out


def render_rows(render: Path, secs, mask) -> list[dict]:
    st = render / "stems"
    voc = _load(st / "lead_dry.wav")
    voc = np.repeat(voc, 2, axis=0) if voc.shape[0] == 1 else voc
    for p in sorted(st.glob("wet_*.wav")):
        w = _load(p)
        voc = voc[:, : w.shape[1]] + w[:, : voc.shape[1]]
    inst = _load(st / "inst_ducked.wav")
    return per_section(frame_k_energy(voc), frame_k_energy(inst), lambda j: j, secs, mask)


def main() -> int:
    from mix_automation.v8.match import SONG_REFS, TARGETS, cover_name, load_drift_curve, orig_to_cover

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--songs", nargs="*", default=list(SONG_REFS))
    ap.add_argument("--render", nargs="*", default=[], help="v8 render dirs, same order as --songs")
    ap.add_argument("--all-lead", action="store_true", help="lead-active frames (ignore non-lead stems)")
    a = ap.parse_args()
    res: dict = {"songs": {}}
    rel_rows, sec_rows = [], []
    for i, song in enumerate(a.songs):
        main_m, lead_m = main_only_mask(song)
        mask = lead_m if a.all_lead else main_m
        secs = sections(song)
        pros = {}
        for pair in SONG_REFS[song]:
            curve = load_drift_curve(pair)
            ev = frame_k_energy(_load(TARGETS / pair / "target_vocal.wav"))
            ei = frame_k_energy(_load(TARGETS / pair / "inst_pro.wav"))

            def f(j, curve=curve):
                return np.round(orig_to_cover(j * FRAME + FRAME / 2, curve) / FRAME - 0.5).astype(int)

            pros[cover_name(pair)] = per_section(ev, ei, f, secs, mask)
        renders = {}
        if song in R5_RENDERS:
            renders["R5"] = render_rows(ROOT / "data" / "outputs_v8r5" / "render" / R5_RENDERS[song], secs, mask)
        if i < len(a.render):
            renders["new"] = render_rows(Path(a.render[i]), secs, mask)
        pt = {c: by_type(r) for c, r in pros.items()}
        med = {t: (float(np.median([v[t] for v in pt.values() if v[t] is not None]))
                   if any(v[t] is not None for v in pt.values()) else None) for t in TYPES}
        # rel per cover first, then median (cover-level pairing removes each cover's overall offset)
        rel = {}
        for t in TYPES:
            vals = [v[t] - v["verse"] for v in pt.values() if v[t] is not None and v["verse"] is not None]
            rel[t] = float(np.median(vals)) if vals else None
        rt = {k: by_type(r) for k, r in renders.items()}
        res["songs"][song] = {"pro_by_type": pt, "pro_median": med, "pro_rel": rel, "renders_by_type": rt,
                              "main_only_s": float(main_m.sum() * FRAME / SR), "lead_s": float(lead_m.sum() * FRAME / SR),
                              "sections": [{**{k: v for k, v in r.items() if k not in ("sv", "si")},
                                            "pro": {c: pros[c][j]["kvi"] for c in pros},
                                            **{k: renders[k][j]["kvi"] for k in renders}}
                                           for j, r in enumerate(next(iter(pros.values())))]}
        for t in TYPES:
            rr = {k: (v[t] - v["verse"] if v[t] is not None and v["verse"] is not None else None) for k, v in rt.items()}
            rel_rows.append([song, t, len(pt), _f(med[t]), _f(rel[t]),
                             "; ".join(f"{k} {_f(v[t])} / rel {_f(rr[k])}" for k, v in rt.items()) or "–"])
        for s in res["songs"][song]["sections"]:
            pv = [v for v in s["pro"].values() if v is not None]
            sec_rows.append([song[:6], s["label"], s["type"], f"{s['frames_s']:.0f}", _f(np.median(pv) if pv else None),
                             _f(s.get("R5")), _f(s.get("new"))])
    pooled = {}
    for t in TYPES:
        vals = [d["pro_rel"][t] for d in res["songs"].values() if d["pro_rel"][t] is not None]
        pooled[t] = float(np.median(vals)) if vals else None
    res["pooled_rel"] = pooled
    res["mask"] = "lead_active" if a.all_lead else "main_only"
    print_table(["song", "section", "type", "s", "pro med", "R5", "new"], sec_rows)
    print_table(["song", "type", "n", "pro kVI", "pro rel", "renders"], rel_rows)
    print("pooled rel (median over songs): " + ", ".join(f"{t} {_f(v)}" for t, v in pooled.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name(OUT.stem + ("_alllead" if a.all_lead else "") + ".json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"json: {out}")
    return 0


def _f(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:+.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
