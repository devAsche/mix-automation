"""Phase 2c: human pro covers of the same four songs as references.

Stages (run in order; every stage caches under data/processed/v8_cover_refs/, nothing is written
into youtube_references/):

    ingest    webm -> 44.1 kHz wav -> BS-RoFormer (same model as orig_stems / v8_v7_master_sep)
    identity  separated cover inst vs the official off-vocal used by v7: global offset, envelope
              correlation, windowed offsets (tempo drift / edits), chroma key shift, length
    stats     metrics of the covers on the v7 time axis -> docs/evaluations/cover_stats.json
              (summary statistics only; the audio is never committed)

    .venv\\Scripts\\python.exe scripts\\analysis\\cover_refs.py ingest|identity|stats|all

evaluate_issues.py reads cover_stats.json for the global rules, the leave-one-out false-alarm rate,
the phrase comparison (eva-2, eva-4) and the step negatives (eva-g4, six-3, six-8).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_ch_first, load_report, print_table  # noqa: E402
from _v8eval import _cached, _clean, _fid, _key, analyze_stem_pair  # noqa: E402
from build_reference_stats import V7_DIRS, scalars  # noqa: E402

from mix_automation.v8.metrics import loudness  # noqa: E402
from mix_automation.v8.metrics._core import (  # noqa: E402
    band_power,
    db,
    p56_activity,
    phrases,
    stft_power,
    to_mono,
)

SR = 44100
REFS = ROOT / "youtube_references" / "歌った曲references"
CACHE = ROOT / "data" / "processed" / "v8_cover_refs"
WAV = CACHE / "wav"
STEMS = CACHE / "stems"
ALIGNED = CACHE / "aligned"
ARCHIVE = ROOT / "data" / "archive" / "gemini_v7_20261126"
V7SEP = ROOT / "data" / "processed" / "v8_v7_master_sep"
OUT = ROOT / "docs" / "evaluations" / "cover_stats.json"
FOLDERS = {
    "Eva_HiiragiKirai": "エバ",
    "FallInLove_Suiet": "Fall in Love",
    "LoveAndDestroy_MI8K": "ラブ&デストロイ",
    "SixFeetUnder_Aq3ra": "シックスフィートアンダー",
}
ENV_HOP = 441  # 10 ms
MAX_LAG_S = 90.0
# Local search +-0.15 s: below half a beat at these tempi (0.21-0.32 s), so a window cannot lock onto the
# neighbouring beat of a periodic backing (seen with +-2 s: spurious "drift" on a cover with envelope corr 0.99).
WIN_S, WIN_HOP_S, LOCAL_SEARCH_S = 20.0, 10.0, 0.15
# Identity criteria. Envelope correlation of two renders of the same backing after alignment is
# ~0.8-0.95 (separated inst vs official off-vocal of the same song, orig_stems: see identity table);
# an unrelated arrangement stays below ~0.3 (the threshold already used by
# reference_extractor._align_and_check_key_shift, aligned_ok >= 0.30).
CORR_SAME = 0.5
CORR_UNRELATED = 0.3
# Tempo: a drift of the local offset by more than one 10 ms envelope hop per minute, or an edit
# (local offset off the linear fit by more than 0.2 s = about a 16th note at 140 BPM), marks the cover.
DRIFT_MS_PER_MIN = 10.0
EDIT_S = 0.2
# A cover must span at least 80 % of the v7 time axis to enter the song-level ranges.
MIN_COVERAGE = 0.8


def items() -> list[dict]:
    out = []
    for folder, song in FOLDERS.items():
        for f in sorted((REFS / folder).glob("*.webm")):
            kind, _, rest = f.stem.partition("_")
            name = rest.rsplit("_", 1)[0] if kind == "cover" else "original"
            out.append({"song": song, "folder": folder, "kind": kind, "name": name, "src": f,
                        "slug": f"{folder.split('_')[0]}__{kind}_{name}"})
    return out


def v7_dir(song: str) -> Path:
    return ARCHIVE / V7_DIRS[song]


def offvocal_path(song: str) -> Path:
    return v7_dir(song) / "Media" / "00_offvocal_raw_backup.wav"


# --------------------------------------------------------------------------- ingest


def ingest() -> None:
    from mix_automation.analysis.reference_extractor import BSRoFormerSeparator

    WAV.mkdir(parents=True, exist_ok=True)
    sep = BSRoFormerSeparator(cache_dir=STEMS, model_dir=ROOT / "data" / "processed" / "roformer_models")
    for it in items():
        wav = WAV / f"{it['slug']}.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(it["src"]), "-ac", "2", "-ar", str(SR),
                            "-c:a", "pcm_s16le", str(wav)], check=True)
        sep.separate(wav, it["slug"], target_sr=SR)
        print(f"separated {it['slug']}", flush=True)


def stem_paths(it: dict) -> tuple[Path, Path]:
    return STEMS / f"{it['slug']}_vocals.wav", STEMS / f"{it['slug']}_inst.wav"


# --------------------------------------------------------------------------- identity


def _env(x: np.ndarray) -> np.ndarray:
    k = len(x) // ENV_HOP
    e = np.log10(np.sqrt(np.mean(x[: k * ENV_HOP].reshape(k, ENV_HOP) ** 2, axis=1)) + 1e-5)
    return e


def _z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


def _best_lag(ref: np.ndarray, x: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    """Lag L (frames) maximising corr(ref[t], x[t + L]) for L in [lo, hi], normalised per overlap."""
    from scipy.signal import correlate, correlation_lags

    c = correlate(_z(x), _z(ref), mode="full", method="fft")
    lags = correlation_lags(len(x), len(ref), mode="full")
    ov = np.minimum(len(ref), len(x) - lags) - np.maximum(0, -lags)
    sel = (lags >= lo) & (lags <= hi) & (ov > 0.3 * min(len(ref), len(x)))
    if not sel.any():
        return 0, 0.0
    s = c[sel] / ov[sel]
    i = int(np.argmax(s))
    return int(lags[sel][i]), float(s[i])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def shift_to(x: np.ndarray, lag: int, n: int) -> np.ndarray:
    """y[..., t] = x[..., t + lag] for t in [0, n), zero outside."""
    out = np.zeros(x.shape[:-1] + (n,), dtype=np.float32)
    a, b = max(0, -lag), min(n, x.shape[-1] - lag)
    if b > a:
        out[..., a:b] = x[..., a + lag : b + lag]
    return out


def key_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    """Semitones by which b must be rotated to match a (CQT chroma), best and zero-shift correlation."""
    import librosa

    ca = librosa.feature.chroma_cqt(y=a.astype(np.float32), sr=SR, hop_length=4096)
    cb = librosa.feature.chroma_cqt(y=b.astype(np.float32), sr=SR, hop_length=4096)
    m = min(ca.shape[1], cb.shape[1])
    ca, cb = ca[:, :m], cb[:, :m]
    act = (ca.sum(0) > 1e-3) & (cb.sum(0) > 1e-3)
    ca, cb = ca[:, act], cb[:, act]
    ca = ca - ca.mean(0, keepdims=True)
    rs = []
    for k in range(12):
        r = np.roll(cb, k, axis=0)
        r = r - r.mean(0, keepdims=True)
        rs.append(float(np.sum(ca * r) / (np.linalg.norm(ca) * np.linalg.norm(r) + 1e-12)))
    k = int(np.argmax(rs))
    return (k if k <= 6 else k - 12), rs[k], rs[0]


def identity_of(it: dict) -> dict:
    vp, ip = stem_paths(it)
    off = offvocal_path(it["song"])
    key = _key("cover_identity", 3, _fid(ip), _fid(off))

    def run() -> dict:
        ci = to_mono(load_ch_first(ip))
        om = to_mono(load_ch_first(off))
        ec, eo = _env(ci), _env(om)
        mx = int(MAX_LAG_S * SR / ENV_HOP)
        lag, _ = _best_lag(eo, ec, -mx, mx)
        # Windowed local offsets around the global one (off-vocal time axis).
        w, h, s = (int(v * SR / ENV_HOP) for v in (WIN_S, WIN_HOP_S, LOCAL_SEARCH_S))
        loc = []
        for a in range(0, max(1, len(eo) - w + 1), h):
            ref = eo[a : a + w]
            seg_lo, seg_hi = a + lag - s, a + lag + w + s
            if seg_lo < 0 or seg_hi > len(ec) or ref.std() < 1e-3:
                continue
            seg = ec[seg_lo:seg_hi]
            ll, cc = _best_lag(ref, seg, 0, 2 * s)
            corr = _pearson(ref, ec[a + lag - s + ll : a + lag - s + ll + w]) if a + lag - s + ll + w <= len(ec) else np.nan
            loc.append([(a + w / 2) * ENV_HOP / SR, (lag - s + ll) * ENV_HOP / SR, corr])
        loc_a = np.array(loc, dtype=float) if loc else np.zeros((0, 3))
        good = loc_a[np.isfinite(loc_a[:, 2]) & (loc_a[:, 2] >= CORR_SAME)] if len(loc_a) else loc_a
        drift = edits = None
        if len(good) >= 3:
            pf = np.polyfit(good[:, 0], good[:, 1], 1)
            drift = float(pf[0] * 60_000.0)  # ms per minute
            edits = int(np.sum(np.abs(good[:, 1] - np.polyval(pf, good[:, 0])) > EDIT_S))
        n = len(eo)
        ec_al = shift_to(ec[None, :], lag, n)[0]
        ov = np.zeros(n, dtype=bool)
        ov[max(0, -lag) : max(0, min(n, len(ec) - lag))] = True
        corr = _pearson(eo[ov], ec_al[ov]) if ov.sum() > 100 else 0.0
        # Key: chroma on the aligned overlap (up to 120 s from the start of the overlap).
        ci_al = shift_to(ci[None, :], lag * ENV_HOP, len(om))[0]
        nz = np.flatnonzero(np.abs(ci_al) > 0)
        k_st, k_best, k_zero = (0, 0.0, 0.0)
        if len(nz) > SR * 5:
            a0 = nz[0]
            a1 = min(nz[-1], a0 + 120 * SR)
            k_st, k_best, k_zero = key_shift(om[a0:a1], ci_al[a0:a1])
        coverage = float(ov.sum() / n)
        return {
            "offset_s": lag * ENV_HOP / SR, "env_corr": corr, "coverage": coverage,
            "len_s": len(ci) / SR, "offvocal_len_s": len(om) / SR,
            "drift_ms_per_min": drift, "edits": edits, "n_windows": int(len(loc_a)),
            "local_corr_median": float(np.nanmedian(loc_a[:, 2])) if len(loc_a) else None,
            "key_shift": k_st, "chroma_best": k_best, "chroma_zero": k_zero,
        }

    return _cached(f"cover_identity_{it['slug']}", key, run)


def status_of(idn: dict) -> tuple[str, str]:
    """'use' / 'flag' / 'exclude' and why."""
    why = []
    if idn["env_corr"] < CORR_UNRELATED:
        return "exclude", f"オフボと相関 {idn['env_corr']:.2f} < {CORR_UNRELATED}（別アレンジ）"
    if idn["key_shift"] != 0:
        why.append(f"移調 {idn['key_shift']:+d}")
    if idn["drift_ms_per_min"] is not None and abs(idn["drift_ms_per_min"]) > DRIFT_MS_PER_MIN:
        why.append(f"テンポずれ {idn['drift_ms_per_min']:+.0f} ms/分")
    if idn["edits"]:
        why.append(f"編集 {idn['edits']} 窓")
    if idn["env_corr"] < CORR_SAME:
        why.append(f"相関 {idn['env_corr']:.2f}")
    if idn["coverage"] < MIN_COVERAGE:
        return "exclude", f"v7 の時間軸の {idn['coverage'] * 100:.0f}% のみ（{idn['len_s']:.0f} s）" + ("; " + ", ".join(why) if why else "")
    if any(w.startswith(("テンポ", "編集")) for w in why):
        return "exclude", ", ".join(why)
    return ("flag", ", ".join(why)) if why else ("use", "")


# --------------------------------------------------------------------------- stats


def aligned_stems(it: dict, idn: dict) -> tuple[Path, Path]:
    """Cover stems shifted onto the v7 (off-vocal) time axis, same length as the off-vocal."""
    ALIGNED.mkdir(parents=True, exist_ok=True)
    vp, ip = stem_paths(it)
    av, ai = ALIGNED / f"{it['slug']}_vocals.wav", ALIGNED / f"{it['slug']}_inst.wav"
    lag = int(round(idn["offset_s"] * SR))
    stamp = ALIGNED / f"{it['slug']}.lag"
    if not (av.exists() and ai.exists() and stamp.exists() and stamp.read_text() == str(lag)):
        import soundfile as sf

        n = int(round(idn["offvocal_len_s"] * SR))
        for src, dst in ((vp, av), (ip, ai)):
            x = load_ch_first(src)
            sf.write(str(dst), shift_to(x, lag, n).T, SR, subtype="PCM_16")
        stamp.write_text(str(lag))
    return av, ai


def v7_sep_paths(song: str) -> tuple[Path, Path]:
    d = V7_DIRS[song]
    return V7SEP / f"v7master_{d}_vocals.wav", V7SEP / f"v7master_{d}_inst.wav"


def phrase_table(vocal_p: Path, inst_p: Path, segs: list[list[float]], name: str) -> list[float | None]:
    """Partial-loudness masking loss (phon) of the vocal in each given segment (v7 phrases)."""
    from mix_automation.v8.metrics._core import Segment

    key = _key("cover_phr", 1, _fid(vocal_p), _fid(inst_p), json.dumps(segs))

    def run() -> dict:
        v, i = load_ch_first(vocal_p), load_ch_first(inst_p)
        n = min(v.shape[1], i.shape[1])
        tr = loudness.partial_loudness_track(v[:, :n], i[:, :n], SR)
        ph = loudness.phrase_loudness(tr, [Segment(a, b) for a, b in segs])
        # Only frames where this vocal is active count; a cover that does not sing the phrase -> None.
        _, act, times = phrases(v[:, :n], SR)
        out = []
        for p, (a, b) in zip(ph, segs, strict=True):
            sel = (times >= a) & (times < b)
            share = float(act[: len(times)][sel].mean()) if sel.any() else 0.0
            out.append(p.masking_loss_phon if share >= 0.3 else None)
        return {"loss": out}

    return _cached(f"cover_phr_{name}", key, run)["loss"]


def boundary_steps(vocal_p: Path, inst_p: Path, times_s: list[float], name: str, win_s: float = 3.0) -> list[dict]:
    """Raw step of the vocal at the given times: Δlevel = Δ(vocal − backing) dB, Δbright = Δ(2-8 kHz −
    150 Hz-1 kHz) of the vocal (same bands as steps.py), mean over vocal-active frames win_s after −
    before. Raw = includes singing, arrangement and mix changes (no pre vocal exists for a cover)."""
    key = _key("cover_steps", 1, _fid(vocal_p), _fid(inst_p), json.dumps(times_s), win_s)

    def run() -> dict:
        v, i = load_ch_first(vocal_p), load_ch_first(inst_p)
        n = min(v.shape[1], i.shape[1])
        fv, tv, pv = stft_power(v[:, :n], SR)
        _, _, pi = stft_power(i[:, :n], SR)
        m = min(pv.shape[1], pi.shape[1])
        lev = np.asarray(db(band_power(pv, fv, 100.0, SR / 2)))[:m] - np.asarray(db(band_power(pi, fv, 100.0, SR / 2)))[:m]
        br = (np.asarray(db(band_power(pv, fv, 2000.0, 8000.0))) - np.asarray(db(band_power(pv, fv, 150.0, 1000.0))))[:m]
        act, _, _ = p56_activity(to_mono(v[:, :n]), SR)
        act, tv = act[:m], tv[:m]
        rows = []
        for t in times_s:
            b = (tv >= t - win_s) & (tv < t) & act
            a = (tv >= t) & (tv < t + win_s) & act
            if b.sum() < 10 or a.sum() < 10:
                rows.append({"t": t, "d_level": None, "d_bright": None})
                continue
            rows.append({"t": t, "d_level": float(lev[a].mean() - lev[b].mean()), "d_bright": float(br[a].mean() - br[b].mean())})
        return {"rows": rows}

    return _cached(f"cover_steps_{name}", key, run)["rows"]


def step_times() -> dict[str, dict[str, float]]:
    """Boundary times under test: six-3 / six-8 at the far_score peak of the D metric, eva-g4 at the
    |Δbright| >= JND peaks that sit on v7 section boundaries (as in evaluate_issues.judge_global_step)."""
    from _v8eval import analyze_debug_render, analyze_render
    from evaluate_issues import LEVEL_JND, PHASE2_OUT

    out: dict[str, dict[str, float]] = {}
    for song in ("エバ", "シックスフィートアンダー"):
        d = PHASE2_OUT / (V7_DIRS[song].split("_auto")[0] + "_auto")
        d = d if (d / "debug_stems" / "lead_dry.wav").exists() else v7_dir(song)
        res = analyze_render(d)
        dbg = analyze_debug_render(d)
        dd = dbg["D_dry"] if dbg and dbg.get("D_dry") else res["D"]
        t = np.array(dd["t"], dtype=float)
        br = np.array([np.nan if x is None else x for x in dd["d_bright"]], dtype=float)
        far = np.array([np.nan if x is None else x for x in res["D"]["far"]], dtype=float)
        tf = np.array(res["D"]["t"], dtype=float)
        bounds = [x["start_sec"] for x in (load_report(d).get("sections") or [])][1:]
        if song == "シックスフィートアンダー":
            for iid, t0 in (("six-3", 53.0), ("six-8", 163.0)):
                w0, w1 = (t0 - 3.0, t0 + 6.0) if iid == "six-3" else (t0 - 1.0, t0 + 4.0)
                sel = (tf >= w0) & (tf <= w1) & np.isfinite(far)
                out[iid] = {iid: float(tf[sel][np.argmax(far[sel])])}
        else:
            order = np.argsort(-np.nan_to_num(np.abs(br), nan=-1))
            peaks = []
            for k in order:
                if not np.isfinite(br[k]) or abs(br[k]) < LEVEL_JND:
                    break
                if all(abs(t[k] - u) >= 3.0 for u in peaks):
                    peaks.append(float(t[k]))
            on_b = sorted(u for u in peaks if any(abs(u - b) <= 1.5 for b in bounds))
            out["eva-g4"] = {f"eva-g4@{u:.1f}": u for u in on_b}
    return out


def stats() -> None:
    import yaml

    issues = yaml.safe_load((ROOT / "docs" / "evaluations" / "issues.yaml").read_text(encoding="utf-8"))["issues"]
    its = items()
    ident, songs = {}, {}
    for it in its:
        idn = identity_of(it)
        st, why = status_of(idn)
        ident[it["slug"]] = {"song": it["song"], "kind": it["kind"], "name": it["name"], "status": st, "why": why} | idn
    st_times = step_times()
    for song in FOLDERS.values():
        v7v, v7i = v7_sep_paths(song)
        v7 = analyze_stem_pair(f"v7sep_{V7_DIRS[song]}", v7v, v7i)
        v7_phr = [[p[0], p[1]] for p in v7["A"]["phrases"]]
        refs = {}
        for it in its:
            if it["song"] != song:
                continue
            idn = ident[it["slug"]]
            if idn["status"] == "exclude" and idn["coverage"] < 0.05:
                continue
            av, ai = aligned_stems(it, idn)
            r = analyze_stem_pair(f"cover_{it['slug']}", av, ai, with_inst_rt=False)
            refs[it["slug"]] = {
                "kind": it["kind"], "status": idn["status"], "scalars": scalars(r),
                "phrase_loss": phrase_table(av, ai, v7_phr, it["slug"]),
            }
            print(f"stats {it['slug']}", flush=True)
        times = {k: v for d in st_times.values() for k, v in d.items() if (song == "エバ") == k.startswith("eva")}
        steps_rows = {}
        if times:
            tl = list(times.values())
            steps_rows["v7"] = boundary_steps(v7v, v7i, tl, f"v7sep_{V7_DIRS[song]}")
            for slug, rr in refs.items():
                if rr["kind"] == "cover" and rr["status"] != "exclude":
                    av, ai = ALIGNED / f"{slug}_vocals.wav", ALIGNED / f"{slug}_inst.wav"
                    steps_rows[slug] = boundary_steps(av, ai, tl, slug)
        songs[song] = {
            "v7_sep": scalars(v7),
            "v7_phrases": v7_phr,
            "v7_phrase_loss": [p[2] for p in v7["A"]["phrases"]],
            "refs": refs,
            "step_times": times,
            "steps": steps_rows,
        }
    out = {
        "schema_version": 1,
        "description": "Phase 2c: human pro covers (YouTube webm, BS-RoFormer separated, aligned to the v7 off-vocal). "
        "scripts/analysis/cover_refs.py stats",
        "criteria": {"corr_same": CORR_SAME, "corr_unrelated": CORR_UNRELATED, "drift_ms_per_min": DRIFT_MS_PER_MIN,
                     "edit_s": EDIT_S, "min_coverage": MIN_COVERAGE},
        "identity": ident,
        "songs": songs,
        "issues_used": [i["id"] for i in issues if i["song"] in FOLDERS.values()],
    }
    OUT.write_text(json.dumps(_clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


def show_identity() -> None:
    rows = []
    for it in items():
        idn = identity_of(it)
        st, why = status_of(idn)
        rows.append([it["song"][:6], f"{it['kind']}_{it['name']}", f"{idn['len_s']:.0f}", f"{idn['offset_s']:+.2f}",
                     f"{idn['env_corr']:.2f}", f"{idn['local_corr_median'] or 0:.2f}", f"{idn['coverage'] * 100:.0f}%",
                     "—" if idn["drift_ms_per_min"] is None else f"{idn['drift_ms_per_min']:+.1f}", idn["edits"],
                     f"{idn['key_shift']:+d} ({idn['chroma_best']:.2f}/{idn['chroma_zero']:.2f})", st, why])
    print_table(["曲", "音源", "長さs", "offset s", "env相関", "窓相関中央", "被覆", "drift ms/分", "編集", "移調(best/0)", "判定", "理由"], rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ingest", "identity", "stats", "all"])
    a = ap.parse_args()
    if a.stage in ("ingest", "all"):
        ingest()
    if a.stage in ("identity", "all"):
        show_identity()
    if a.stage in ("stats", "all"):
        stats()


if __name__ == "__main__":
    main()
